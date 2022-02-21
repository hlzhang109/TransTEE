The code is based on [CausalM](https://github.com/amirfeder/CausaLM)
## POMS Gender/Race Experimental Pipeline

### Prerequisites
- Create the CausaLM conda environment: `conda env create --file causalm_gpu_env.yml`
- Install the [`en_core_web_lg`](https://spacy.io/models/en#en_core_web_lg) spaCy model.
- Download the *gender* and *race* [datasets](https://www.kaggle.com/amirfeder/causalm) and place them in the `./datasets` folder.
- Make sure the `CAUSALM_DIR` variable in `constants.py` is set to point to the path where the CausaLM datasets are located.


### training and test
- `POMS_GendeRace/pipeline/training.py --treatment <gender/race> --model <"transtee", "drnet", "tarnet", "vcnet">`

This will train and test all the POMS causal estimators for the full experimental pipeline for Gender or Race treatment.
