# Fintech-Final HCCR

- Handwritten Chinese Character Recognition
- This repo is revised from [Offline Handwritten Chinese Character Classifier](https://github.com/pavlo-melnyk/offline-HCCR).

## Hierachy
```
├───src
│   ├───melnyk_net.py
│   ├───predictor.py
│   └─── ....
├───training_data
│   ├───combine (training_src)
│   └───combine_similar (training_src)
└───models
    ├───models_1 (log_dir)
    │   ├───models_1_1.hdf5
    │   ├───models_1_2.hdf5
    │   └─── ....
    └─── ....
```

## Train
```
time python melnyk_net.py --log_dir models_1 --training_src combine
```

## Inference Single Image
```
python predictor.py --img_path images/巧_YMmNG5.jpg --model_path model_combine.hdf5
```

## Model Performance
| Model File                    | Best Valid Accuracy |    #Classes          |
| ----------------------------- | ------------------- | -------------------- |
| `model_combine.hdf5`          | 0.9268              | 801                  |
| `models_combine_similar.hdf5` | 0.9192              | 879                  |

## Contact
If you have any question, please feel free to contact me by sending email to [r08946014@ntu.edu.tw](mailto:r08946014@ntu.edu.tw)