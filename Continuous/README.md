
The code is based on [VCNet](https://github.com/lushleaf/varying-coefficient-net-with-functional-tr)

# How to run

## Synthetic Dataset

-- generate the synthetic dataset

    datas/simu1_generate_data.py

-- train and evaluating the methods

To run a singe run of models/methods with one dataset. You can also use it to generate estimated ADRF curve :
    
    main_batch.py --plt_adrf

To run all models/methods with numbers of datasets, please use

    main_batch.py

For regularization terms TR/PTR, use  --p 1/2, 1 for TR, 2 for PTR

    main_batch.py --p [1, 2]

## IHDP Dataset

-- generate the IHDP dataset

    datas/ihdp_generate_data.py

-- train and evaluating the methods

To run a singe run of models/methods with one dataset. You can also use it to generate estimated ADRF curve :
    
    main_batch_ihdpv2.py --plt_adrf

To run all models/methods with numbers of datasets, please use

    main_batch_ihdpv2.py

For regularization terms TR/PTR, use  --p 1/2, 1 for TR, 2 for PTR

    main_batch_ihdpv2.py --p [1, 2]

## News Dataset

-- generate the synthetic dataset
    datas/news_process.py
    datas/news_generate_data.py

-- train and evaluating the methods

To run a singe run of models/methods with one dataset. You can also use it to generate estimated ADRF curve :
    
    main_batch_newsv2.py  --plt_adrf

To run all models/methods with numbers of datasets, please use

    main_batch_newsv2.py

For regularization terms TR/PTR, use  --p 1/2, 1 for TR, 2 for PTR

    main_batch_ihdpv2.py --p [1, 2]
