import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataframe = pd.read_csv("ex_2_metric/dataset.csv")

nb_peoples = len(dataframe.index)

#compute the dissimilarity matrix
def compute_dissimilarity(person_1_id, person_2_id):

    person_1_age = dataframe.loc[person_1_id][1]
    person_2_age = dataframe.loc[person_2_id][1]
    
    person_1_job = dataframe.loc[person_1_id][3]
    person_2_job = dataframe.loc[person_2_id][3]

    person_1_city = dataframe.loc[person_1_id][4]
    person_2_city = dataframe.loc[person_2_id][4]
    
    person_1_music = dataframe.loc[person_1_id][5]
    person_2_music = dataframe.loc[person_2_id][5]

    
    
    if ((person_1_job ==person_2_job)):
        dissimilarity_job = 0
    elif ((person_1_job in ["designer", "painter"])and (person_2_job in ["designer", "painter"])):
        dissimilarity_job = 0.5
    elif ((person_1_job in ["developper", "engineer"]) and (person_2_job in ["developper", "engineer"])):
        dissimilarity_job = 0.5
    else :
        dissimilarity_job = 1

    if (person_1_city == person_2_city):
        dissimilarity_city = 0
    else:
        dissimilarity_city = 1

    if (person_1_music == person_2_music):
        dissimilarity_music = 0
    elif ((person_1_music in ["rock", "metal", "technical death metal"]) and (person_2_music in ["rock", "metal", "technical death metal"])):
        dissimilarity_music = 0.5
    elif ((person_1_music in ["hiphop", "rap"]) and (person_2_music in ["hiphop", "rap"])):
        dissimilarity_music = 0.5
    else:
        dissimilarity_music = 1

    # we build a hybrid dissimilarity
    dissimilarity = math.sqrt(
        (person_1_age - person_2_age) ** 2
        + dissimilarity_job * 1.3
        + dissimilarity_city * 1.7
        + dissimilarity_music * 1.7
    )
    return dissimilarity


# build a dissimilarity matrix
dissimilarity_matrix = np.zeros((nb_peoples, nb_peoples))
print("compute dissimilarities")
for person_1_id in range(nb_peoples):
    for person_2_id in range(nb_peoples):
        dissimilarity = compute_dissimilarity(person_1_id, person_2_id)
        dissimilarity_matrix[person_1_id, person_2_id] = dissimilarity

print("--- dissimilarity matrix ---")
print(dissimilarity_matrix)

print ("--- mean ---")
print (np.mean(dissimilarity_matrix))

print ("--- standard deviation ---")
print (np.std(dissimilarity_matrix))
np.save("dissimilarity.npy", dissimilarity_matrix)
