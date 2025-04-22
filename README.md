<h1>cat_drug_synergy</h1>

<h2>Abstract</h2>

<p>
This repository contains the code for the paper: <br>
<a href="https://www.biorxiv.org/content/10.1101/2024.06.12.598611v3">
Optimizing drug synergy prediction through categorical embeddings in Deep Neural Networks
</a>.
</p>

<p>
We present a machine learning pipeline for <strong>predicting drug synergy</strong> using <strong>categorical embeddings</strong> in deep neural networks.
By representing categorical variables (e.g., drug identity, cell line type) as dense vectors, this approach improves predictive accuracy on sparse pharmacological datasets.
Evaluated on the <strong>AstraZeneca-Sanger DREAM Challenge</strong> dataset, our method outperforms traditional encoding techniques across multiple models,
providing a scalable and efficient tool for combination drug discovery.
</p>

<hr>

<h2>Installation</h2>

<pre><code>git clone https://github.com/edmanft/cat_drug_synergy.git
cd cat_drug_synergy
pip install -r requirements.txt
</code></pre>

<hr>

<h2>Quick Start</h2>

<ol>
<li><strong>Preprocess data</strong>:</li>
<pre><code>from src.data.process_data import load_dataset, split_dataset

full_dataset_df, column_type_dict = load_dataset(drug_syn_path, cell_lines_path, drug_portfolio_path)
datasets = split_dataset(full_dataset_df)
</code></pre>

<li><strong>Train and evaluate Pytorch Tabular model</strong>:</li>
<pre><code>from src.model.evaluation import weighted_pearson, train_evaluate_pytorch_tabular_pipeline


eval_dict, trained_model, training_time = train_evaluate_pytorch_tabular_pipeline(
                datasets=datasets,
                data_config=data_config,
                model_config=model_config,
                trainer_config=trainer_config,
                verbose=args.verbose, 
                seed = args.seed)
</code></pre>
<li><strong>Save trained Pytorch Tabular model to reuse embedding</strong>:</li>

<pre><code>if args.model_dir is not None:
                model_save_path = os.path.join(args.model_dir, f"{model_name}_model.ckpt")
                trained_model.save_model(model_save_path)
</code></pre>

<li><strong>Use the embedding to encode categorical variables</strong>:</li>
<pre><code>from src.model.evaluation import train_evaluate_sklearn_pipeline

eval_dict = train_evaluate_sklearn_pipeline(
                datasets=datasets,
                model=model,
                categorical_encoder=args.encoder, 
                model_path=args.model_path,
                verbose=False)
</code></pre>
</ol>

<hr>

<h2>Dataset</h2>

<ul>
<li>Source: <strong>AstraZeneca-Sanger Drug Combination DREAM Challenge</strong>.</li>
<li>3,475 experiments with 167 drug pairs, 85 cancer cell lines.</li>
<li>Features: numerical (IC50, Hill slope, E_inf), categorical (drug names, targets, pathways, cell lines).</li>
<li>Target: <strong>Synergy score</strong> (Loewe model).</li>
</ul>

<hr>

<h2>License</h2>

<p>Apache 2.0 License.</p>

<hr>

<h2>Citation</h2>

<pre><code>@article{gonzalez2025drugsynergy,
  title={Optimizing drug synergy prediction through categorical embeddings in Deep Neural Networks},
  author={Gonzalez Lastre, Manuel and Gonzalez de Prado Salas, Pablo and Guantes, Raul},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.06.12.598611v3}
}
</code></pre>
