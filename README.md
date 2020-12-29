# Multilabel classification of skill sets based on a job description.
Enseirb-Matmeca, Bordeaux INP | [Anas AHOUZI](https://www.linkedin.com/in/anas-ahouzi-6aab0b155/)
***

## :monocle_face: Description
- This project aims to predict a set of skills for a particular job based on its description. The user enters the job title alongside the description, and
gets a set of skills as an output for this job. The model is trained to predict around 123 different skills. </br>

- The skills predicted by the model involve the following elements: Abilities, skills, technology skills, tools used, work context,
 work style. You can directly have access to the excel sheet files used for creating the dataset by clicking on this [link](https://drive.google.com/drive/folders/11OtcCNTCFuVbsHZsuWqeQnF1L6-lsgc-?usp=sharing).
 
- **The most challenging part** in this project is to construct the labels for the model. By processing the various excel files,
we find at least 4000 labels, that can either be continious or categorical. Therefore, various statistical techniques were used in order
to reduce the label space dimensionality (Pearson, Spearman, Cramer's V).


## :rocket: Repository Structure
The repository contains the following files & directories:
- **Dataset directory:** It contains a data pre-processing notebook where the final dataset used for training 
the model is constructed from various excel sheet files.
- **Features directory:** Implementation of various functions used for cleaning the text, and encoding it.
- **Train directory:** Implementation of the GRU model, and training/evaluation process.


---
## :mailbox_closed: Contact
For any information, feedback or questions, please [contact me][anas-email]










[anas-email]: mailto:ahouzi2000@hotmail.fr