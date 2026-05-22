
Description

This project presents an AI-driven framework for the early detection of mental health issues among university students in academic environments. The study introduces a hybrid ensemble learning architecture designed to identify symptoms of depression, anxiety, and panic attacks using structured demographic, academic, and self-reported psychological data. Built on a publicly available student mental health dataset, the system integrates both classical machine learning and deep learning approaches to improve predictive accuracy and generalization. The work contributes to the growing field of digital mental health analytics by providing a scalable and interpretable solution that can support proactive mental health monitoring in higher education institutions.

Key contributions of the project include:
- A HybridDL-ML-Ensemble model that combines Random Forest, XGBoost, Multilayer Perceptron (MLP), and CNN‑1D within a stacked learning framework, with logistic regression used as a meta‑learner for final prediction.
- A comprehensive data preprocessing and feature engineering pipeline that transforms survey responses, demographic attributes, and academic indicators into machine‑learning‑ready representations.
- Robust evaluation using stratified cross‑validation, achieving high predictive performance with accuracy of 0.9915, precision of 0.9875, recall of 0.9936, and F1‑score of 0.9902.
- Interpretability and deployment considerations, including feature importance analysis and a transparent meta‑learning layer suitable for real‑world institutional use.

The proposed framework can be applied in university counseling systems, learning management platforms, or digital health monitoring tools to help identify at‑risk students earlier. By transforming routinely collected academic and survey data into actionable insights, the system supports timely intervention, better resource allocation, and improved student well‑being in educational settings.

Dataset Information

The study utilizes the Student Mental Health dataset, a publicly available survey-based dataset designed to analyze psychological well-being among university students. The dataset contains demographic, academic, and self-reported mental health indicators that enable machine learning models to predict early signs of psychological distress.
Dataset Name | Source | Dataset Type | Scale & Characteristics | Purpose and Evaluation | LinkStudent Mental Health | Kaggle | Survey-based tabular dataset collected via a Google Form questionnaire | Contains 101 responses from university students. Features include demographic attributes (age, gender, marital status), academic information (course of study, year of study, CGPA range), and mental health indicators (binary responses for depression, anxiety, and panic attacks). Additional information includes whether the student sought specialist treatment. The dataset is structured for supervised learning and includes categorical and numerical attributes suitable for feature engineering and classification tasks. | Used to develop and evaluate AI-based models for early detection of mental health issues among university students. The study formulates a multi-class prediction task (0–3 mental health conditions) based on aggregated self-reported indicators. Model performance is evaluated using Accuracy, Precision, Recall, and F1-score, with results validated through 5-fold stratified cross-validation. | https://kaggle.com/datasets/shariful07/student-mental-health
Student Mental Health | Kaggle | Survey-based tabular dataset collected via a Google Form questionnaire | Includes 101 anonymized student survey records with structured demographic, academic, and psychological variables. Mental health indicators are recorded as binary responses and combined to form a composite mental health score ranging from 0 (no issues) to 3 (depression, anxiety, and panic attack). The dataset contains no missing values because all survey fields were mandatory. | Serves as the primary training and evaluation dataset for the HybridDL-ML-Ensemble model. It supports supervised classification experiments assessing student mental health risk levels. Model effectiveness is measured using macro-averaged classification metrics (Accuracy, Precision, Recall, and F1-score) and analyzed through cross-validation and statistical validation procedures. | https://www.kaggle.com/datasets/shariful07/student-mental-health

- Student Mental Health — 5 None. DATA AVAILABILITY STATEMENT 706 Thedatausedtosupportthefindingsofthisstudyareavailablefreelyat: 707 https://www.kaggle.com/datasets/shariful07/student-mental-health REFE...
- Student Mental Health — . FUNDING 705 None. DATA AVAILABILITY STATEMENT 706 Thedatausedtosupportthefindingsofthisstudyareavailablefreelyat: 707 https://www.kaggle.com/datasets/shariful07/student-mental...

Code Information
Code File | Functionalitymain.py | Main execution script that orchestrates the full pipeline including data loading, preprocessing, model training, cross-validation, ensemble stacking, and evaluation.
data_preprocessing.py | Implements dataset cleaning, binary encoding of survey responses, one-hot encoding of categorical variables, CGPA numerical transformation, feature normalization, and construction of the target variable (mental_health_issues_total).
feature_engineering.py | Generates derived features such as aggregated mental health issue counts and prepares structured feature vectors for model input.
models/random_forest_model.py | Defines the Random Forest classifier configuration with 200 estimators and Gini impurity criterion, used as one of the base learners in the ensemble.
models/xgboost_model.py | Implements the XGBoost classifier with gradient boosting trees, configured with 200 estimators, learning rate 0.1, and maximum depth of 6.
models/mlp_model.py | Implements the Multilayer Perceptron neural network with two hidden layers (128 and 64 neurons), ReLU activation, and softmax output for multi-class prediction.
models/cnn1d_model.py | Defines the 1D Convolutional Neural Network architecture including convolution, ReLU activation, max-pooling, and dense layers for extracting local feature interactions.
ensemble/stacking_ensemble.py | Implements the HybridDL-ML-Ensemble stacking framework that combines predictions from Random Forest, XGBoost, MLP, and CNN-1D into meta-features.
ensemble/meta_learner.py | Implements the logistic regression meta-learner that receives concatenated probability outputs from base models and produces the final prediction.
training/cross_validation.py | Handles stratified 5-fold cross-validation, leakage-free training of base models, generation of out-of-fold predictions, and construction of the meta-dataset.
evaluation/metrics.py | Computes performance metrics including Accuracy, Precision, Recall, and F1-score using macro-averaging across classes.
evaluation/statistical_tests.py | Performs statistical analyses such as paired t-tests, Wilcoxon signed-rank tests, and permutation testing to validate model significance.
visualization/plots.py | Generates visualizations such as boxplots, radar charts, z-score distributions, ranking heatmaps, confusion matrices, and comparative metric charts.
external_validation.py | Evaluates the trained ensemble model on external datasets such as PHQ-9 and UCI Student Performance datasets to assess generalization capability.
Usage Instructions
Clone and Set Up the Environment

Clone the project repository and create a Python environment for running the experiments.

git clone https://github.com/your-username/hybrid-mental-health-ensemble.git
cd hybrid-mental-health-ensemble

Create and activate a virtual environment.

CPU:
python -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

Install required dependencies.

pip install -r requirements.txt

Typical dependencies include:
- numpy  
- pandas  
- scikit-learn  
- xgboost  
- tensorflow  
- matplotlib  
- seaborn  

GPU users should install TensorFlow with GPU support.

GPU:
pip install tensorflow

CPU-only:
pip install tensorflow-cpu

Prepare Data

Download the dataset Student Mental Health from Kaggle:
- Student Mental Health  
  https://kaggle.com/datasets/shariful07/student-mental-health

or
- Student Mental Health  
  https://www.kaggle.com/datasets/shariful07/student-mental-health

Download and place the dataset file inside the project data directory:

mkdir data
mv Student_Mental_Health.csv data/

Run the preprocessing pipeline to prepare the dataset for training.

python preprocess.py --input data/Student_Mental_Health.csv --output data/processed.csv

The preprocessing pipeline performs the following steps:
- Convert binary responses (Yes/No) to numeric values (0/1)  
- One-hot encode categorical variables (gender, marital status, course)  
- Convert CGPA ranges to numerical midpoints  
- Normalize numerical features such as age and CGPA using Min-Max scaling  
- Create the target label mental_health_issues_total (0–3) by summing depression, anxiety, and panic attack indicators  
- Remove non-informative features such as timestamps  

Train the Model

Training uses a stacking ensemble consisting of:
- Random Forest  
- XGBoost  
- Multilayer Perceptron (MLP)  
- 1D Convolutional Neural Network (CNN-1D)  
- Logistic Regression meta-learner  

Run the training script with 5-fold stratified cross-validation.

CPU:
python train.py \
  --data data/processed.csv \
  --cv 5 \
  --rf_estimators 200 \
  --xgb_estimators 200 \
  --xgb_lr 0.1 \
  --xgb_depth 6 \
  --mlp_hidden 128 64 \
  --epochs 100 \
  --batch_size 32

GPU (recommended for deep learning models):
python train.py \
  --data data/processed.csv \
  --cv 5 \
  --rf_estimators 200 \
  --xgb_estimators 200 \
  --xgb_lr 0.1 \
  --xgb_depth 6 \
  --mlp_hidden 128 64 \
  --epochs 100 \
  --batch_size 32 \
  --device gpu

Training process:
Split the dataset using stratified 5-fold cross-validation.  
Train base models (RF, XGBoost, MLP, CNN-1D) on each fold.  
Generate out-of-fold predictions from base models.  
Concatenate prediction probabilities to form meta-features.  
Train the logistic regression meta-learner on these meta-features.  
Retrain base models on the full dataset for final deployment.

Saved outputs:
- trained base models  
- meta-learner model  
- training metrics and logs  

Evaluate and Run Inference

Evaluate the trained model on validation folds.

python evaluate.py \
  --data data/processed.csv \
  --model_dir models/

The evaluation script reports:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

To run inference on new student records:

python predict.py \
  --model_dir models/ \
  --input sample_student.csv \
  --output predictions.csv

Example input format (sample_student.csv):
- age  
- gender  
- marital_status  
- course  
- year_of_study  
- cgpa  
- depression  
- anxiety  
- panic_attack  

The inference pipeline:
Applies the same preprocessing used during training.  
Generates probability predictions from each base model.  
Concatenates probabilities into meta-features.  
Uses the logistic regression meta-learner to produce the final prediction.  

Output prediction:
- Class 0: No reported mental health issues  
- Class 1: One reported condition  
- Class 2: Two reported conditions  
- Class 3: Three reported conditions

Requirements
- Python ≥ 3.9  
- NumPy ≥ 1.23  
- Pandas ≥ 1.5  
- scikit-learn ≥ 1.2  
- XGBoost ≥ 1.7  
- TensorFlow ≥ 2.10  
- Matplotlib ≥ 3.6  
- Seaborn ≥ 0.12  
- SciPy ≥ 1.9  
- SHAP ≥ 0.41 (optional, for model interpretability)

Methodology

This study proposes a hybrid stacked learning framework, termed HybridDL-ML-Ensemble, to enable early detection of mental health issues among university students. The methodology integrates both deep learning and classical machine learning models to exploit their complementary strengths. Tree-based models effectively capture structured feature interactions in tabular data, while neural networks learn nonlinear patterns and latent feature relationships. The ensemble architecture combines four base learners—Random Forest, XGBoost, Multilayer Perceptron (MLP), and a 1D Convolutional Neural Network (CNN‑1D)—whose predictions are aggregated through a logistic regression meta-learner.

The input dataset consists of demographic, academic, and psychological survey features. After preprocessing and feature engineering, the feature vector is passed to each base model independently. Each model generates a class probability vector representing the likelihood that a student belongs to one of four mental health severity categories (0–3 reported conditions). These outputs are concatenated to form a meta-feature representation used by the second-level classifier. The logistic regression meta-learner then produces the final prediction by learning optimal weights for the base-model outputs.

Training is performed using stratified five-fold cross-validation to preserve class distribution across folds and to avoid information leakage. In each fold, base models are trained on the training subset, and their out-of-fold predictions are used to construct meta-features for the validation subset. This process ensures that the meta-learner only observes predictions generated from unseen data. During inference, the trained base models generate prediction probabilities for a new sample, which are concatenated and passed to the meta-learner to produce the final classification.

Network Architecture

The neural components of the framework consist of an MLP and a CNN‑1D model designed to capture nonlinear feature relationships and local feature interactions from the structured tabular input. The CNN‑1D architecture follows an encoder–decoder style structure composed of a contracting path for feature extraction and an expanding path for representation refinement prior to classification.

Contracting Path

The contracting path serves as the feature extraction stage of the CNN‑1D model. The input feature vector is first reshaped into a one-dimensional sequence so that convolutional operations can learn local interactions among neighboring attributes. The sequence is passed through a convolutional layer with multiple filters (64 filters with kernel size 3). Each filter scans across the input vector to detect patterns representing relationships among demographic, academic, and psychological variables.

After convolution, a Rectified Linear Unit (ReLU) activation introduces nonlinearity, enabling the network to capture complex dependencies between features. A max-pooling layer with pool size 2 follows the activation layer, reducing the spatial dimension of the feature maps while preserving the most informative responses. This downsampling step decreases computational complexity and encourages the network to learn increasingly abstract representations. Through convolution and pooling, the contracting path gradually compresses the input representation while extracting high-level feature descriptors relevant to mental health prediction.

Expanding Path

The expanding path reconstructs and refines the compressed feature representation produced by the contracting path. The pooled feature maps are flattened and passed through a fully connected dense layer with 64 neurons. This stage integrates information extracted from different convolutional filters and allows interactions between distant features that may not have been captured during convolution.

The dense representation is further processed by nonlinear activation functions, enabling the network to combine local patterns into global predictive signals. Finally, a softmax classification layer generates a probability distribution across the four target classes representing the number of mental health conditions reported. The expanding path therefore transforms compressed convolutional features into a structured decision space suitable for classification.

Integration with Ensemble Framework

Outputs from the CNN‑1D model, along with predictions from the MLP, Random Forest, and XGBoost models, form probability vectors for each sample. These vectors are concatenated into a meta-feature vector. A multinomial logistic regression model serves as the meta-learner, analyzing consensus and disagreement among the base models to produce the final prediction. This stacked architecture improves robustness by combining diverse learning paradigms and reducing individual model biases.

Results Summary

The proposed HybridDL-ML-Ensemble model was evaluated using a five-fold stratified cross-validation framework on a dataset of 101 university students. Performance was measured using four standard classification metrics: Accuracy, Precision, Recall, and F1-score. The results demonstrate that the ensemble framework consistently outperforms individual base learners and traditional machine learning classifiers.

Fold-wise Performance
Fold | Accuracy | Precision | Recall | F1-score1 | 0.9901 | 0.9862 | 0.9928 | 0.9895
2 | 0.9923 | 0.9884 | 0.9945 | 0.9914
3 | 0.9910 | 0.9871 | 0.9932 | 0.9901
4 | 0.9908 | 0.9869 | 0.9935 | 0.9900
5 | 0.9933 | 0.9890 | 0.9940 | 0.9910
The aggregated performance across folds is:
- Accuracy: 0.9915 ± 0.0012
- Precision: 0.9875 ± 0.0011
- Recall: 0.9936 ± 0.0007
- F1-score: 0.9902 ± 0.0008

Weighted metrics considering class distribution further confirm consistent performance:
- Weighted Accuracy: 0.9915
- Weighted Precision: 0.9881
- Weighted Recall: 0.9915
- Weighted F1-score: 0.9898

Ablation Study

To evaluate the contribution of different model components, three configurations were compared: a machine-learning-only ensemble, a deep-learning-only ensemble, and the proposed hybrid framework.
Model Configuration | Accuracy | Precision | Recall | F1-scoreML-only (RF + XGBoost) | 0.9628 | 0.9581 | 0.9652 | 0.9616
DL-only (MLP + CNN-1D) | 0.9689 | 0.9643 | 0.9710 | 0.9676
Hybrid Ensemble (Proposed) | 0.9915 | 0.9875 | 0.9936 | 0.9902
The hybrid configuration significantly improves predictive performance by combining the strengths of both classical and deep learning models.

Comparison with Classical Machine Learning Models

The proposed model was also compared with traditional classifiers trained under the same preprocessing and evaluation settings.
Model | Accuracy | Precision | Recall | F1-scoreSupport Vector Machine | 0.921 | 0.913 | 0.925 | 0.919
Logistic Regression | 0.906 | 0.897 | 0.915 | 0.906
K-Nearest Neighbors | 0.878 | 0.865 | 0.889 | 0.876
HybridDL-ML-Ensemble (Proposed) | 0.991 | 0.987 | 0.994 | 0.990
The hybrid ensemble achieves a substantial performance improvement over conventional models, demonstrating its effectiveness in identifying mental health risks.

External Validation Results

To assess generalization capability, the model was evaluated on two additional datasets with different feature structures and label definitions.
Dataset | Sample Size | Data Type | Accuracy | Precision | Recall | F1-scorePHQ-9 Depression Dataset (Zenodo) | 157 | Clinical Labels (PHQ-9) | 0.936 | 0.921 | 0.936 | 0.928
UCI Student Performance Dataset | 395 | Behavioral Proxy Labels | 0.902 | 0.881 | 0.917 | 0.899
The results indicate strong adaptability across heterogeneous datasets, maintaining high predictive performance even under different label definitions and feature spaces.

Statistical Significance Analysis

Permutation testing with 1000 iterations produced a null accuracy distribution with mean 0.7423 and standard deviation 0.0417. The observed accuracy (0.9915) lies far outside this distribution, resulting in:
- Empirical p-value: p < 0.001

Additional statistical tests confirm the significance of improvements over baseline models.
Comparison | Paired t-test (p-value) | Wilcoxon (p-value)Hybrid vs CNN-1D | 0.0021 | 0.0047
Hybrid vs Random Forest | < 0.01 | < 0.01
Hybrid vs XGBoost | < 0.01 | < 0.01
These results demonstrate that the performance gains of the HybridDL-ML-Ensemble model are statistically significant and robust across multiple evaluation criteria.

Citations

References
Bielinski, A., Rojek, I., and Mikolajewski, D. (2023). Comparison of selected machine learning algorithms in the analysis of mental health indicators. Electronics, 12. doi:10.3390/electronics12214407  
Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.  
Chancellor, S., and De Choudhury, M. (2020). Methods in predictive techniques for mental health status on social media: A critical review. NPJ Digital Medicine, 3, 43.  
Chen, M., and Jiang, S. (2019). Analysis and research on mental health of college students based on cognitive computing. Cognitive Systems Research, 56, 151–158. doi:10.1016/j.cogsys.2019.03.003  
Chen, T., He, T., Benesty, M., Khotilovich, V., Tang, Y., Cho, H., et al. (2015). XGBoost: Extreme gradient boosting. R package version 0.4-21, 1–4.  
Danner, M., Hadzic, B., Gerhardt, S., Ludwig, S., Uslu, I., Shao, P., et al. (2023). Advancing mental health diagnostics: GPT-based method for depression detection. In Proceedings of the 62nd Annual Conference of the Society of Instrument and Control Engineers (SICE), 1290–1296. doi:10.23919/SICE59929.2023.10354236  
Diao, F., and Xia, D. (2025). A deep learning framework based on CNN and LSTM for monitoring college students’ psychological states. In Proceedings of the 10th International Conference on Cyber Security and Information Engineering, 205–210.  
Dong, G., Tang, M., Cai, L., Barnes, L. E., and Boukhechba, M. (2021). Semi-supervised graph instance transformer for mental health inference. In 20th IEEE International Conference on Machine Learning and Applications (ICMLA 2021), 1221–1228. doi:10.1109/ICMLA52953.2021.00198  
Elsayed, N., ElSayed, Z., Asadizanjani, N., Ozer, M., Abdelgawad, A., and Bayoumi, M. (2022). Speech emotion recognition using supervised deep recurrent system for mental health monitoring. In 2022 IEEE 8th World Forum on Internet of Things (WF-IoT). doi:10.1109/WF-IOT54382.2022.10152117  
Ghrouz, A. K., Noohu, M. M., Manzar, M. D., Spence, D. W., BaHammam, A. S., and Pandi-Perumal, S. R. (2019). Physical activity and sleep quality in relation to mental health among college students. Sleep and Breathing, 23, 627–634. doi:10.1007/s11325-019-01780-z  
Kiranyaz, S., Avci, O., Abdeljaber, O., Ince, T., Gabbouj, M., and Inman, D. J. (2021). 1D convolutional neural networks and applications: A survey. Mechanical Systems and Signal Processing, 151, 107398.  
Lattie, E. G., Cohen, K. A., Hersch, E., Williams, K. D. A., Kruzan, K. P., MacIver, C., et al. (2022). Uptake and effectiveness of a self-guided mobile app platform for college student mental health. Internet Interventions, 27. doi:10.1016/j.invent.2021.100493  
Lin, Y. S., Tai, L. K., and Chen, A. L. (2023). The detection of mental health conditions by incorporating external knowledge. Journal of Intelligent Information Systems, 61, 497–518.  
Lin, Y. S., Tai, L. K., and Chen, A. L. P. (2023). The detection of mental health conditions by incorporating external knowledge. Journal of Intelligent Information Systems, 61, 497–518. doi:10.1007/s10844-022-00774-w  
Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.  
Ma, S., Yang, J., Xu, J., Zhang, N., Kang, L., Wang, P., et al. (2022). Using network analysis to identify central symptoms of college students’ mental health. Journal of Affective Disorders, 311, 47–54. doi:10.1016/j.jad.2022.05.065  
Nissen, L. R., Tsamardinos, I., Eskelund, K., Gradus, J. L., Andersen, S. B., and Karstoft, K.-I. (2021). Forecasting military mental health in a complete sample of Danish military personnel deployed between 1992–2013. Journal of Affective Disorders, 288, 167–174. doi:10.1016/j.jad.2021.04.010  
Qeadan, F., Madden, E. F., Barbeau, W. A., Mensah, N. A., Azagba, S., and English, K. (2022). Associations between discrimination and adverse mental health symptoms and disorder diagnoses among college students in the United States. Journal of Affective Disorders, 310, 249–257. doi:10.1016/j.jad.2022.05.026  
Ranjan, R., Neeti, and Sahana, B. C. (2022). Automatic detection of mental health status using alpha subband of EEG data. In 2022 IEEE International Symposium on Medical Measurements and Applications (MeMeA). doi:10.1109/MEMEA54994.2022.9856586  
Shen, X. (2023). Data mining-based innovative model for mental health of college students using IoT and big data analysis. Soft Computing, 27, 14483–14495. doi:10.1007/s00500-023-09083-y  
Shen, X. (2023). Data mining-based innovative model for mental health of college students using IoT and big data analysis. Soft Computing, 27, 14483–14495.  
Sun, Y., Li, H., Wu, H., and Fu, Y. (2021). Machine learning-based mental health analysis and early warning for college student. In 2021 International Conference on Software Quality, Reliability and Security Companion (QRS-C), 569–578. doi:10.1109/QRS-C55045.2021.00087  
Tasci, G., Gun, M. V., Keles, T., Tasci, B., Barua, P. D., Tasci, I., et al. (2023). QLBP: Dynamic patterns-based feature extraction functions for automatic detection of mental health and cognitive conditions using EEG signals. Chaos, Solitons & Fractals, 172. doi:10.1016/j.chaos.2023.113472  
Taud, H., and Mas, J.-F. (2017). Multilayer perceptron (MLP). In Geomatic Approaches for Modeling Land Change Scenarios. Springer, 451–455.  
Tlachac, M. L., Toto, E., Lovering, J., Kayastha, R., Taurich, N., and Rundensteiner, E. (2021). EMU: Early mental health uncovering framework and dataset. In 20th IEEE International Conference on Machine Learning and Applications (ICMLA 2021), 1311–1318. doi:10.1109/ICMLA52953.2021.00213  
Tyagi, A., Singh, V. P., and Gore, M. M. (2023). Towards artificial intelligence in mental health: A comprehensive survey on the detection of schizophrenia. Multimedia Tools and Applications, 82, 20343–20405. doi:10.1007/s11042-022-13809-9  
Wang, Y., and Zhao, X. (2025). CASTLE: A multi-modal educational data fusion framework for student mental health detection using MOON network embedding and deep neural networks. Informatica, 49.  
Xie, W., Wang, C., Lin, Z., Luo, X., Chen, W., Xu, M., et al. (2022). Multimodal fusion diagnosis of depression and anxiety based on CNN-LSTM model. Computerized Medical Imaging and Graphics, 102, 102128.  
Yang, L., Ni, H., and Zhu, Y. (2025). Data-driven mental health assessment of college students using ES-ANN and LOF algorithms during public health events. Informatica, 49.  
Zhang, Z. (2024). Early warning model of adolescent mental health based on big data and machine learning. Soft Computing, 28, 1567–1584. doi:10.1007/s00500-023-09422-z  
Zhang, Z. (2024). Early warning model of adolescent mental health based on big data and machine learning. Soft Computing, 28, 811–828.

License

No explicit license information was provided in the supplied document. Unless otherwise stated by the authors or publisher, all rights to the content remain with the respective authors and/or the publishing entity. Users should consult the original publication source or publisher for official licensing terms and permissions regarding reuse, distribution, or modification of the material.

Contribution Guidelines

Contributions should focus on improving, extending, or validating the AI-based framework for early detection of mental health issues among university students. Researchers and developers are encouraged to contribute in ways that enhance model robustness, interpretability, and real-world applicability in academic environments.

Scope of Contributions
Contributions may include, but are not limited to:
- Improvements to the HybridDL-ML-Ensemble architecture or its individual components (Random Forest, XGBoost, MLP, CNN-1D, or the logistic regression meta-learner).
- Enhancements to preprocessing pipelines, feature engineering methods, or data normalization strategies.
- Integration of additional data modalities such as behavioral logs, wearable sensor data, or social media signals.
- Implementation of explainable AI techniques (e.g., SHAP, LIME) to improve interpretability of predictions.
- Optimization techniques that reduce computational complexity while maintaining predictive performance.
- Replication studies, benchmarking experiments, or validation on new datasets from diverse academic institutions.

Data and Ethical Considerations
All contributions involving datasets must respect privacy and ethical standards:
- Use anonymized or publicly available datasets.
- Avoid including personally identifiable information (PII).
- Ensure that mental health data are handled responsibly and used only for research and decision-support purposes.
- Clearly document data sources, preprocessing steps, and any labeling or proxy construction methods.

Code and Experiment Reproducibility
To ensure transparency and reproducibility:
- Provide clear documentation for code, dependencies, and environment setup.
- Include scripts or notebooks for preprocessing, model training, and evaluation.
- Report experimental settings such as hyperparameters, cross-validation procedures, and evaluation metrics.
- Maintain leakage-free training procedures when implementing stacking or ensemble learning methods.

Model Evaluation
All contributions introducing model modifications should include:
- Evaluation using standard classification metrics such as Accuracy, Precision, Recall, and F1-score.
- Cross-validation experiments to demonstrate stability across data splits.
- Comparative analysis with baseline models where applicable.
- Statistical validation when performance improvements are claimed.

Documentation and Reporting
Contributors should provide concise descriptions of their changes, including:
- The motivation and expected benefits of the contribution.
- Technical details of algorithms or architectural modifications.
- Experimental results and limitations.

Clear documentation ensures that future researchers and institutions can effectively adopt and extend the framework for mental health monitoring in academic settings.


