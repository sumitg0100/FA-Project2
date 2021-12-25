import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler


from yellowbrick.target import FeatureCorrelation
from scipy.stats import norm
from scipy import stats


import warnings
warnings.filterwarnings("ignore")
color = sns.color_palette()
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


data = pd.read_csv("C:\\Users\\Sumit\\Desktop\\Sumit Gupta\\MBA Study Material\\VSIP\\data_o.csv")

df_artist = pd.read_csv("C:\\Users\\Sumit\\Desktop\\Sumit Gupta\\MBA Study Material\\VSIP\\artists.csv")

df_by_genres = pd.read_csv("C:\\Users\\Sumit\\Desktop\\Sumit Gupta\\MBA Study Material\\VSIP\\data_by_genres_o.csv")

df_year = pd.read_csv("C:\\Users\\Sumit\\Desktop\\Sumit Gupta\\MBA Study Material\\VSIP\\data_by_year_o.csv")

df_track = pd.read_csv("C:\\Users\\Sumit\\Desktop\\Sumit Gupta\\MBA Study Material\\VSIP\\tracks.csv")

df_pop_artist = pd.read_csv("C:\\Users\\Sumit\\Desktop\\Sumit Gupta\\MBA Study Material\\VSIP\\data_by_artist_o.csv")


data.describe()

data.info()

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data['popularity']

# Create a list of the feature names

features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()       




total = data.shape[0]
popularity_score_more_than_40 = data[data['popularity'] > 40].shape[0]

probability = (popularity_score_more_than_40/total)*100
print("Probability of song getting more than 40 in popularity :", probability)


use_col = ['acousticness','danceability','loudness','popularity','duration_ms','energy','speechiness','valence']

df_mod = pd.read_csv('C:\\Users\\Sumit\\Desktop\\Sumit Gupta\\MBA Study Material\\VSIP\\data_o.csv', usecols=use_col,nrows=30000)
df_mod.to_csv('file1.csv') 
df_mod.head()

cor = df_mod.corr()
sns.heatmap(cor)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X = df_mod.drop(columns=['popularity'])
y = df_mod['popularity']

x_train,x_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=27)

print("num of  train sample in train set:",x_train.shape)
print("Number of samples in validation set:",y_test.shape)

from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()

random_forest.fit(x_train, y_train)
Y_pred_rf = random_forest.predict(x_test)
random_forest.score(x_train,y_train)
acc_random_forest = round(random_forest.score(x_train,y_train) * 100, 2)

print("Important features")
pd.Series(random_forest.feature_importances_,x_train.columns).sort_values(ascending=True).plot.barh(width=0.8)
print('__'*30)
print(acc_random_forest)

mymodle = open('fordeploy.pkl', 'wb')



import pickle 

pickle.dump(random_forest,mymodle,protocol=pickle.HIGHEST_PROTOCOL)

mymodle.close()

import numpy as np 
import pickle

mymodel = open('fordeploy.pkl', 'rb')

model = pickle.load(mymodel)

data = np.array([0.995,0.708,158648,0.1950,	-12.428,0.0506,0.7790])
data = data.reshape(1,-1)

pre = model.predict(data)
print(pre)


import streamlit as st 
import streamlit.components.v1 as components 

# load the saved model 
pickle_in = open("fordeploy.pkl","rb")
model=pickle.load(pickle_in)


def predict_popu(acousticness,danceability,duration_ms,energy,loudness,speechiness,valence):
    """
    this method is for prediction process 
    takes all the Audio characteristics thtat we used for modelling and returns the prediction 
    """
    prediction=model.predict([[acousticness,danceability,duration_ms,energy,loudness,speechiness,valence]])
    print(prediction)
    return prediction



def main():
    st.title("Spotify songs")
     
    html_temp2 = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h2 style="color:white;text-align:center;">Spotify songsr </h2>
        <h1 style="color:white;text-align:center;">Popularity prediction</h1>
		</div>
		"""
     # a simple html code for heading which is in blue color and we can even use "st.write()" also ut for back ground color i used this HTML ..... 
    #  to render this we use ...
    components.html(html_temp2)
    # components.html() will render the render the 

    components.html("""
                <img src="https://www.tech-recipes.com/wp-content/uploads/2016/02/Spotify.png" width="700" height="150">
                
                """)
    # this is to insert the image the in the wed app simple <imag/> tag in HTML
    
    #now lets get the test input from the user by wed app 
    # for this we can use "st.text_input()" which allows use to get the input from the user 
    
    acousticness = st.text_input("acousticness","Type Here")
    danceability = st.text_input("danceability","Type Here")
    duration_ms = st.text_input("duration_ms","Type Here")
    energy = st.text_input("energy","Type Here")
    loudness = st.text_input("loudness","Type Here")
    speechiness = st.text_input("speechiness","Type Here")
    valence = st.text_input("valence","Type Here")
    result=""
    # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
    # after hitting the button the prediction process will go on and then we print the success message by "st.success()"
    if st.button("Predict"):
        result=predict_popu(acousticness,danceability,duration_ms,energy,loudness,speechiness,valence)
    st.success('The Popularity of the song is {}'.format(result))
    # one more button saying About ...
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        
if __name__=='__main__':
    main()












































