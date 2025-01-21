conda activate cat_drug_synergy

python src/scripts/train_classical_ml.py --data_path data --encoder OneHotEncoder --save_path figures/ms/sklearn_OneHotEncoder.csv
python src/scripts/train_classical_ml.py --data_path data --encoder LabelEncoder --save_path figures/ms/sklearn_LabelEncoder.csv
python src/scripts/train_classical_ml.py --data_path data --encoder EmbeddingEncoder --model_path model_weights/TabTransformer_model.ckpt --save_path figures/ms/sklearn_EmbeddingTabTransformer.csv
python src/scripts/train_classical_ml.py --data_path data --encoder EmbeddingEncoder --model_path model_weights/AutoInt_model.ckpt --save_path figures/ms/sklearn_EmbeddingAutoInt.csv
python src/scripts/train_classical_ml.py --data_path data --encoder EmbeddingEncoder --model_path model_weights/CategoryEmbedding_model.ckpt --save_path figures/ms/sklearn_EmbeddingCategoryEmbedding.csv
python src/scripts/train_classical_ml.py --data_path data --encoder EmbeddingEncoder --model_path model_weights/TabNet_model.ckpt --save_path figures/ms/sklearn_EmbeddingTabNet.csv
python src/scripts/train_pytorch_tabular.py --data_path data --batch_size 512 --max_epoch 200 --es_patience 5 --model_dir models_weights --save_path figures/ms/pytorch_tabular.csv