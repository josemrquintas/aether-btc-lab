"""Tests for GA chromosome — Phase 3 GA Optimization Gate."""

import pytest

from aether_btc.ga.chromosome import GENE_SPECS, Chromosome


class TestRandomChromosome:
    """Test random chromosome generation."""

    def test_random_chromosome_within_bounds(self):
        """All genes within min/max after random init."""
        for _ in range(10):
            c = Chromosome.random()
            for spec in GENE_SPECS:
                val = c.get(spec.name)
                assert spec.min_val <= val <= spec.max_val, (
                    f"{spec.name}: {val} not in [{spec.min_val}, {spec.max_val}]"
                )

    def test_random_chromosomes_differ(self):
        """Two random chromosomes should not be identical."""
        c1 = Chromosome.random()
        c2 = Chromosome.random()
        assert c1.genes != c2.genes


class TestMutation:
    """Test mutation stays in bounds."""

    def test_mutation_stays_in_bounds(self):
        """After mutation, no gene exceeds its range."""
        c = Chromosome.random()
        for _ in range(50):
            mutated = c.mutate(mutation_rate=0.5)
            for spec in GENE_SPECS:
                val = mutated.get(spec.name)
                assert spec.min_val <= val <= spec.max_val, (
                    f"{spec.name}: {val} not in [{spec.min_val}, {spec.max_val}]"
                )

    def test_mutation_changes_genes(self):
        """Mutation actually changes some genes."""
        c = Chromosome.from_defaults()
        mutated = c.mutate(mutation_rate=1.0)  # Mutate all
        changed = sum(1 for name in c.genes if c.genes[name] != mutated.genes[name])
        assert changed > 0


class TestCrossover:
    """Test crossover produces valid children."""

    def test_crossover_produces_valid_children(self):
        """Both children have all genes, all within bounds."""
        p1 = Chromosome.random()
        p2 = Chromosome.random()
        c1, c2 = p1.crossover(p2)

        assert len(c1.genes) == len(GENE_SPECS)
        assert len(c2.genes) == len(GENE_SPECS)

        for spec in GENE_SPECS:
            assert spec.name in c1.genes
            assert spec.name in c2.genes
            # Values should come from one parent
            assert c1.get(spec.name) in (p1.get(spec.name), p2.get(spec.name))
            assert c2.get(spec.name) in (p1.get(spec.name), p2.get(spec.name))

    def test_crossover_mixes_genes(self):
        """Crossover actually mixes genes from both parents."""
        p1 = Chromosome.from_defaults()
        p2 = Chromosome.random()
        c1, _ = p1.crossover(p2)

        from_p1 = sum(1 for name in c1.genes if c1.genes[name] == p1.genes[name])
        from_p2 = sum(1 for name in c1.genes if c1.genes[name] == p2.genes[name])
        # Should have genes from both parents
        assert from_p1 > 0
        assert from_p2 > 0


class TestCryptoGenes:
    """Test crypto-specific genes are present."""

    def test_leverage_genes_present(self):
        """Chromosome has all leverage genes."""
        c = Chromosome.from_defaults()
        leverage_genes = ["leverage_base", "leverage_confidence_scale",
                         "leverage_volatility_dampen", "max_leverage_cap"]
        for gene in leverage_genes:
            assert gene in c.genes

    def test_funding_genes_present(self):
        """Chromosome has funding rate and liquidation buffer genes."""
        c = Chromosome.from_defaults()
        assert "funding_rate_threshold" in c.genes
        assert "liquidation_buffer_pct" in c.genes

    def test_default_chromosome_has_all_genes(self):
        """Default chromosome has all specified genes."""
        c = Chromosome.from_defaults()
        gene_names = {spec.name for spec in GENE_SPECS}
        assert set(c.genes.keys()) == gene_names
