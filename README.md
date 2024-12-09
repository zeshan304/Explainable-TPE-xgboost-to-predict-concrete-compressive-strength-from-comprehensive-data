# About

This repository contains the code and supplementary materials for the research paper titled "**TPE-xgboost for explainable predictions of concrete compressive strength considering compositions, and mechanical and microstructure properties of testing samples**".

## Abstract of Research Paper

The use of machine learning (ML) for predicting concrete compressive strength (CCS) has shown promising and accurate results, making it a valuable tool in the field. However, efficient prediction requires not only a robust ML approach but also a comprehensive and well-curated dataset. This study addresses this challenge by investigating an extensive and integrated dataset of (1) concrete compositions (mixture proportions) and (2) testing conditions (mechanical and microstructure properties of concrete testing samples), containing 1525 observations relating to 39 parameters. On algorithms side, this research proposes novel tree-structured parzen estimator based extreme gradient boosting (TPE-xgboost) for accurate and confident CCS predictions. Moreover, SHapley Additive exPlanations (SHAP) analysis was performed to provide a comprehensive understanding of feature importance, dependencies, and interactions. Compared to prior research, this study demonstrates significant improvements in CCS prediction accuracy for separate investigations on concrete compositions (3.77%) and mechanical and microstructure properties (28.57%), while maintaining reasonable accuracy for the integrated dataset despite its complexity and sparseness. SHAP analysis reveals key influential factors, including age and water-to-binder ratio in concrete composition, and peak load and height of testing samples, as well as microstructural characteristics such as mean values of global autocorrelation length and its integral range. This research provides valuable insights into model optimization, predictive performance, and feature dependencies and interactions, contributing to the development of more accurate and reliable CCS prediction models.

### Prerequisites

To set up the required Python environment, you can use the `requirements.txt` file provided in this repository. And to install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```
### Code Structure

The repository contains following four juypter files 

- [`pdf_to_csv_working.ipynb`](pdf_to_csv_working.ipynb): The code to extract and process the data of mechanical and microstructure properties of testing samples from .pdf format to .csv file
- [`df1_working.ipynb`](https://github.com/zeshan304/Explainable-TPE-xgboost-to-predict-concrete-compressive-strength-from-comprehensive-data/blob/main/df1_working.ipynb): The code for anlaysis and modeling of concrete composition data
- [`df2_working.ipynb`](https://github.com/zeshan304/Explainable-TPE-xgboost-to-predict-concrete-compressive-strength-from-comprehensive-data/blob/main/df2_working.ipynb): The code for anlaysis and modeling of mechanical and microstructure properties of testing samples data
- [`df3_working.ipynb`](https://github.com/zeshan304/Explainable-TPE-xgboost-to-predict-concrete-compressive-strength-from-comprehensive-data/blob/main/df3_working.ipynb): The code for anlaysis and modeling of integrated data

## Citation

If you use this code or find the research paper helpful, please consider citing:
```
@article{akber2024tpe,
  title={TPE-xgboost for explainable predictions of concrete compressive strength considering compositions, and mechanical and microstructure properties of testing samples},
  author={Akber, Muhammad Zeshan and Anwar, Ghazanfar Ali and Chan, Wai-Kit and Lee, Hiu-Hung},
  journal={Construction and Building Materials},
  volume={457},
  pages={139398},
  year={2024},
  publisher={Elsevier}
}
```
## Contact
**Muhammad Zeshan Akber**
Email: [zeshan304@gmail.com](mailto:zeshan304@gmail.com)

