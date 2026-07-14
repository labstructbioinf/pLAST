config = {
    # Temporary directory for intermediate files
    "tmp_dir": "tmp",
    # Models and embeddings
    "models": {
        "mmseqs2_model": {
            "type": "mmseqs2",
            "model": "data/models/pscopemmseqs2i03c08_by-level_i03c08_mc2_ep30_vs64_w4_sg0_neg15.model",
            "embeddings": "data/models/pscopemmseqs2i03c08_by-level_i03c08_mc2_ep30_vs64_w4_sg0_neg15_blank_equal.plasmid_emb.pkl",
            "mean_embedding": "data/models/pscopemmseqs2i03c08_by-level_i03c08_mc2_ep30_vs64_w4_sg0_neg15.mean_emb.pkl",
            "idf": "data/models/pscope_mmseqs2_i03_c08_plasmid_archs_perm.idf.pkl",
            "cluster_mapping": "data/export/i03_c08_plasmid_protein_cluster_map.tsv.gz",
            "cluster_token_prefix": "i03c08:",
            "mmseqs_db": "data/mmseqs2/plasmid_protein_representatives",
            "mmseqs_db_padded": "data/mmseqs2/plasmid_protein_representatives_padded",
        },
        "eggnog_model": {
            "type": "eggnog",
            "model": "data/models/refseqeggnog_mc2_ep30_vs64_w4_sg0_neg15.model",
            "embeddings": "data/models/refseqeggnog_mc2_ep30_vs64_w4_sg0_neg15.plasmid_emb.pkl",
            "mean_embedding": "data/models/refseqeggnog_mc2_ep30_vs64_w4_sg0_neg15.mean_emb.pkl",
            "idf": "data/models/refseq_eggnog_plasmid_archs_perm_withnans.idf.pkl",
        },
    },
    # Metadata
    "metadata": "data/plasmid_metadata.tsv.gz",
    "gbfeatures": "data/plasmid_protein_metadata.pkl.gz",

    # eggNOG HMM database for Bacteria
    "hmmscan_db": "data/hmm_db/Bacteria.hmm",
}
