config = {
    # Temporary directory for intermediate files
    "tmp_dir": "tmp",
    # Models and embeddings
    "models": {
        "mmseqs2_model": {
            "type": "mmseqs2",
            "model": "data/mmseqs2i03c08_mc2_ep30_vs64_w4_sg0_neg15.model",
            "embeddings": "data/mmseqs2i03c08_mc2_ep30_vs64_w4_sg0_neg15.emb",
            "mean_embedding": "data/mmseqs2i03c08_mc2_ep30_vs64_w4_sg0_neg15.mean_emb.pkl",
            "idf": "data/mmseqs2i03c08_mc2_ep30_vs64_w4_sg0_neg15.idf.pkl",
        },
        "eggnog_model": {
            "type": "eggnog",
            "model": "data/hits_mc2_ep30_vs64_w4_sg0_neg15.model",
            "embeddings": "data/hits_mc2_ep30_vs64_w4_sg0_neg15.emb",
            "mean_embedding": "data/hits_mc2_ep30_vs64_w4_sg0_neg15.mean_emb.pkl",
            "idf": "data/hits_mc2_ep30_vs64_w4_sg0_neg15.idf.pkl",
        },
    },
    # Metadata
    "metadata": "data/metadata_final.tsv",
    # MMseqs2 clustering data
    "cluster_mapping": "data/circular_plasmids_proteins.tsv",
    "mmseqs_db": "data/circular_plasmids_proteins_representatives",
    "mmseqs_db_padded": "data/circular_plasmids_proteins_representatives_padded",
    # eggNOG HMM database for Bacteria
    "hmmscan_db": "data/Bacteria.hmm",
}
