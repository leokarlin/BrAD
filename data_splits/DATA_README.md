# Download DomainNet
Please download and unzip the DomainNet dataset from: mkdir 

# DomainNet Data Splits
In this directory we keep the text files with the train/test splits used in our paper.
These files where downloaded from https://github.com/zhengzangw/PCS-FUDA/tree/master/data/splits/domainnet for fair comparison
The naming conventions is as follows:
1. the <DOMAIN_NAME> is one of the six domains of DomainNet: real, clipart, painting, sketch, infograph, quickdraw
1. The files <DOMAIN_NAME>_train.txt include all images of the train-set including their labels
1. The files <DOMAIN_NAME>_test.txt include all images of the test-set including their labels
1. The files <DOMAIN_NAME>_train_test.txt include all images of the both the train and test sets. These are the files used in the "Transductive" training. Note that our method does not use any labels during training.
1. The files <DOMAIN_NAME>\_labeled_<SHOT>.txt include the labled examples for the 1shot and 3shot experiments. We use these files during testing as the "Source Domain" 
1. The files <DOMAIN_NAME>\_unlabeled_<SHOT>.txt include the unlabled examples for the 1shot and 3shot experiments. We use these files during testing as the "Destination Domain"

The files include relative paths, the method assumes that the files are next to the image directory:
The file tree should look something like this:
```bash
DomainNet/
├── domain
│   ├── class1
│   ├── class2
│   ├── class3
│   └── class4
├── domain_labeled_1.txt
├── domain_labeled_3.txt
├── domain_test.txt
├── domain_train_test.txt
├── domain_train.txt
├── domain_unlabeled_1.txt
├── domain_unlabeled_3.txt
├── domain2
│   ├── class1
│   ├── class2
│   ├── class3
│   └── class4
├── domain2_labeled_1.txt
├── domain2_labeled_3.txt
├── domain2_test.txt
├── domain2_train_test.txt
├── domain2_train.txt
├── domain2_unlabeled_1.txt
└── domain2_unlabeled_3.txt
```
