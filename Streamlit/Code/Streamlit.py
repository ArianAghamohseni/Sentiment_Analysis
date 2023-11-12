import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import seaborn as ns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score,silhouette_samples
import seaborn as sns
from PIL import Image
from sklearn.cluster import DBSCAN
import time
from sklearn.preprocessing import OneHotEncoder
import mwclient
from imblearn.over_sampling import RandomOverSampler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
from numpy import asarray
from numpy import zeros
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import nltk
from keras.models import Sequential
from keras.utils import to_categorical
# from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline









st.title("Sentiment Analysis")

report_choice = st.sidebar.selectbox("Select a Problem", ["Problem 1", "Problem 2"])


if report_choice == "Problem 1":
    st.header("Problem 1")










































if report_choice == "Problem 2":
    st.header("Problem 2")

    # code

    train = pd.read_csv(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Third Phaze\Streamlit\train_1.csv")
    test = pd.read_csv(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Third Phaze\Streamlit\test_1.csv")
    titles = pd.read_csv(r"D:\Sharif University of Tech\Data Sience Boot Camp\Project\Third Phaze\Streamlit\titles_1.csv")

    # end code

    st.subheader("Choose a Dataset to Display:")
    dataset_choice = st.selectbox("Select a Dataset", ["Train", "Test", "Titles"])

    if dataset_choice == "Train":
        data = train[:100]
    elif dataset_choice == "Test":
        data = test[:100]
    elif dataset_choice == "Titles":
        data = titles[:100]


    st.write(f"Displaying {dataset_choice} Dataset:")
    st.dataframe(data)

    st.subheader("This problem is divided into 3 parts :")
    part_choice = st.selectbox("Which part should we go for ?", ["Part 1", "Part 2", "Part 3"])


    if part_choice == "Part 1":
        st.header("Part 1")
        st.subheader("This part is mostly for understanding the data and familiarizing ourselves with it ")
        
        st.write("""
            We need to handle the Nan values.

            "vote" nans are where there has been no helpful vote for the review, therefore we will fill the nan values in this column with 0s.

            "reviewerName" is not that important in our sentiment analysis and the number of nan data is limited so we won't be needing to handel that.

            "summary" is not pretty important too and because of the number of nans compared to the whole dataset, we will not interfere in this case too.

            "style" won't affect the result much (or at least shouldn't), and thence we will drop the respective column.

            We also have a time column so probably there is no need for "unixTime".
            """)
        
        st.write("Now that it's done, time to check for other data alteration's.")


        st.subheader("Which question should we go for first ?")
        question_num = st.selectbox("Choose the question", ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5", "Question 6", "Furthure analysis"])

        if question_num == "Question 1":
            st.header("First let's see the distribution of overall rating's :")


            # code 

            plt.figure(figsize=(8, 6))
            train['overall'].value_counts().sort_index().plot(kind='bar', edgecolor='black')
            plt.title('Distribution of Overall Ratings')
            plt.xlabel('Overall Rating')
            plt.ylabel('Count')

            # end code 

            st.pyplot(plt)
            st.subheader("""As it is shown, we need to handle the imbalancement.
                We need to make sure that no overfitting is being made, so if we have got overfitting in our final results this is a checkpoint for sure!
                We will use the IMBlearn random oversampler and undersampler for the purpose and in order to make a cautious step overfitting wise, we will bring all the samples to 250000 occurances:""")


        if question_num == "Question 2":
            st.header("After the imbalancement handling, we will go for positive, neutral, and negative wordclouds:")
            
            st.image("word_cloud.png", caption="Initial word cloud", use_column_width=True)   

            st.write("""
                There are some words and verbs in english language that may be repeated in all classes. For instance, "product" is something that can be referred in all classes.

                "need" and "work" are some verbs that have no meaning when used alone. For example you can say: 'This camera works well' and 'This camera doesen't work at all.'.

                There are some other words like these examples that don't add much to our wordcloud's purpose and since we will get rid of them. 
                """)
            
            st.write("""
                As we discussed, we still have some words that are not meant to be there and they are in common between all classes so we will try to delete them too in order to get a better result
                """)


            st.image("word_cloud_finale.png", caption="Fixed word cloud", use_column_width=True)   


            st.subheader("""
                There are still some duplicates like the other wordcloud that we discussed about but as you can see, the positive words in the positive class are much more bold, the negative ones are much more bold in the negative class and both of negative and positive words have been repeated a lot in the neutral class. 
                """)
            
            
        if question_num == "Question 3":
            st.header("For this part let's find the top 10 useful user's")


            # code

            top_reviewers = pd.DataFrame(train.groupby('reviewerID').agg(Mean=('vote', np.mean),Sum=('vote', np.sum))).reset_index()
            top_reviewers['Count'] = top_reviewers["Sum"]/ top_reviewers["Mean"]
            top_reviewers.drop(columns=["Mean"], inplace=True)
            top_reviewers = top_reviewers.sort_values(by='Sum', ascending=False)
            top_10_reviewers = top_reviewers.head(10)
            top_10_reviewer_info = top_10_reviewers.merge(train[['reviewerID', 'reviewerName']], on='reviewerID')
            top_10_reviewer_info = top_10_reviewer_info.drop_duplicates(subset='reviewerName')
            t10df = pd.DataFrame(top_10_reviewer_info[['reviewerName', 'Sum', "Count"]])

            # end code
            st.subheader("Top 10 reviewers with most votes.")

            st.write(t10df)
        

            st.write("""
                In this version you can see in how many comments have the users achieved that much of "useful" votes.

                In other words, "Sum" is how many useful votes have a reviewer recieved and "Count" shows the number of comments they recieved that much attention with.
                """)

        if question_num == "Question 4":
            st.subheader("Lets see the histogram of text length :")


            # code 

            train['text_length'] = train['reviewText'].str.len()

            plt.figure(figsize=(10, 6))
            plt.hist(train['text_length'], bins=30, color='skyblue', edgecolor="black")
            plt.title('Histogram of Text Length (Original)')
            plt.xlabel('Text Length (Number of Characters)')
            plt.ylabel('Frequency')

            # end code 

            st.pyplot(plt)


            st.subheader("Now to find the filter boundaries with the IQR method :")

            # code

            fig = plt.figure(figsize =(10, 7))
            ax = fig.add_axes([0, 0, 1, 1])
            bp = ax.boxplot((train["text_length"]))
            m = np.mean(train["text_length"])
            q3, q1 = np.percentile(train["text_length"], [75 ,25])
            iqr = q3 - q1
            lower = m - 1.5*iqr
            upper = m + 1.5*iqr

            # end code

            st.pyplot(plt)

            st.write("And the bonds would be :")
            st.write((lower, upper))

            st.write("We don't have a -5 characters long word, so we will consider the lower bound as 0.")

            # code 

            filtered_data = train[(train['text_length'] >= 0) & (train['text_length'] <= upper)]

            plt.figure(figsize=(10, 6))
            plt.hist(filtered_data['text_length'], bins=30, color='skyblue', edgecolor="black")
            plt.title('Histogram of Text Length (Filtered)')
            plt.xlabel('Text Length (Number of Characters)')
            plt.ylabel('Frequency')

            # end code 


            st.pyplot(plt)

            st.write("""Based on the analysis of the histogram of text lengths, it seems that setting a constraint on the text length during modeling could be beneficial. 

                        The filtered histogram with a minimum of 0 characters and a maximum of approximately 1250 characters resulted in a smoother distribution of text lengths.

                        Therefore, it is advisable to impose constraints on the text length during modeling.""")
            
            st.write("""After a little explanatory data review you will find out that the min count of characters used in a comment is 200 and therefore our actual interval is (200, 1250) characters.""")


        if question_num == "Question 5":
            st.subheader("First lets see the 10 most five stars :")

            # code 

            t10df = pd.read_csv("t10df.csv")

            # end code


            st.dataframe(t10df)


        if question_num == "Question 6":
            st.subheader("First lets see the 10 most reviewed brands and their score mean :")

            # code 

            top_10_brands = pd.read_csv("top_10_brands.csv")

            # end code


            st.dataframe(top_10_brands)

        if question_num == "Furthure analysis":
            st.subheader("Now we will go for some general and statistical overview on our train data :")
            st.write("""In this view we can see that "overall" is really imbalanced in favor of 5 stars, "vote" is really imbalanced in favor of 0s, and "verified" is really imbalanced in favor of 1s.

                        Also most of our data is gathered before 2017, and our text lenghts vary a lot due to the respective std.""")

            Hypothesis_choice = st.selectbox("Select a Hypothesis", ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3", "Hypothesis 4"])


            if Hypothesis_choice == "Hypothesis 1":
                st.subheader("What are we testing ?")
                st.subheader("Is overall affected by weekday ?")

                st.image("H1.png", caption="over all-frequency bar chart", use_column_width=True)

                st.write("""We already knew that this column isn't normal so we will use U-test for hypothesis testing.

                            First we will generate the data to compare weekdays and weekends and then we will use the U-test to do the Hypothesis testing:

                            X: weekends, Y: weekdays

                            H0 : μx < μy

                            H1 : μx >= μy""")
                

                st.write("And the P_value is : 0.8474923496425621")

                st.write("We can say that people buying on weekdays are more likely to rate the product higher with a pretty good confidence level!")
                st.write("""As you can see in numbers, weekends have a higher mean and a lower std meaning that they are really likely to be greater as in overall rating.

                            (Considering different sample sizes in a U-test)""")






            if Hypothesis_choice == "Hypothesis 2":
                st.subheader("What are we testing ?")
                st.subheader("Is text lenght affected by weekdays ?")

                st.image("H2.png", caption="text length-frequency bar chart", use_column_width=True)

                st.write("""We already knew that this column isn't normal so we will use U-test for hypothesis testing.

                            First we will generate the data to compare weekdays and weekends and then we will use the U-test to do the Hypothesis testing:

                            X: weekends, Y: weekdays

                            H0 : μx < μy

                            H1 : μx >= μy""")
                

                st.write("And the P_value is : 0.0023741145530308443")

                st.write("""As you can see, this hypothesis is rejected due to lack of evidence and therefore we can't come to the result that weekdays comments are longer.

                            Let's see if the opposite is true though:""")
                
                st.write("And the P_value of the opposite test is : 0.9976258855439618")
                
                st.write("""So we can say that comments on weekends are longer with a very vast confidence interval. And it kinda makes sense!""")
                st.write("And it has a greater mean, smaller std.")






            
            if Hypothesis_choice == "Hypothesis 3":
                st.subheader("What are we testing ?")
                st.subheader("Is overall affected by text length ?")

                st.image("H3.png", caption="over all-frequency bar chart", use_column_width=True)

                st.write("""We already knew that this column isn't normal so we will use U-test for hypothesis testing.

                            First we will generate the data to compare weekdays and weekends and then we will use the U-test to do the Hypothesis testing:

                            X: after median, Y: before median

                            H0 : μx < μy

                            H1 : μx >= μy""")
                

                st.write("Due to massive number of outliers, the mean is pretty biased so we will use median to determine which half a comment is.")
                st.write("And it would be : 407.0")

                st.write("And the P_value is : 1.0")

                st.write("So we can say that the more a customer is satisfied with a product, the less thay bother to type long comments!")
                st.write("""Unlike the other hypothesis that we have already discussed this time, the std of the greater class is greaater but the mean difference is so big that the respective bias doesen't afffect the U-test result.""")
                



            if Hypothesis_choice == "Hypothesis 4":
                st.subheader("What are we testing ?")
                st.subheader("Is overall affected by verification status ?")

                st.image("H4.png", caption="over all-frequency bar chart", use_column_width=True)

                st.write("""We already knew that this column isn't normal so we will use U-test for hypothesis testing.

                            First we will generate the data to compare weekdays and weekends and then we will use the U-test to do the Hypothesis testing:

                            X: not-verified, Y: verified

                            H0 : μx < μy

                            H1 : μx >= μy""")
                

                st.write("And the P_value is : 1.0")

                st.write("So we can say that verified comments are generally the ones that have a higher overall score!")





    if part_choice == "Part 2":
        st.subheader("In this part we want to collect the opinions in which discussions about product warranties (such as guarantee, warranty, etc.)")
        choice = st.selectbox("Select a KPI", ["Garante", "Price", "Safety"])

        if choice == "Garante":
            grouped = pd.read_csv("Garante.csv")
            st.subheader("First lets take a look at the data frame :")
            st.dataframe(grouped)

            st.subheader("Top 15 Guarantee Wise:")
            st.dataframe(grouped.head(15))


            st.subheader("Worst 15 Guarantee Wise:")
            st.dataframe(grouped.tail(15))


            st.subheader("And the stacked bar chart :")
            st.image("Garante.png", caption="stacked bar chart", use_column_width=True)   


        if choice == "Price":
            grouped = pd.read_csv("Price.csv")
            st.subheader("The data frame :")
            st.dataframe(grouped)

            st.subheader("Top 15 Price Wise:")
            st.dataframe(grouped.head(15))


            st.subheader("Worst 15 Price Wise:")
            st.dataframe(grouped.tail(15))


            st.subheader("And the stacked bar chart :")
            st.image("Price.png", caption="stacked bar chart", use_column_width=True)   


        if choice == "Safety":
            grouped = pd.read_csv("Safety.csv")
            st.subheader("The data frame per usual :")
            st.dataframe(grouped)

            st.subheader("Top 15 Safety Wise:")
            st.dataframe(grouped.head(15))


            st.subheader("Worst 15 Safety Wise:")
            st.dataframe(grouped.tail(15))


            st.subheader("And the stacked bar chart :")
            st.image("Safety.png", caption="stacked bar chart", use_column_width=True)   


        st.subheader("And we can see these all togheter for better analysis :")
        st.image("Conclusion.png", caption="stacked bar chart", use_column_width=True)  
        












    if part_choice == "Part 3":
        st.header("Part 3")
        st.subheader("This section is about a sentiment analysis with star ranking's.")

        st.write("""We have pretty much processed our data through parts 1&2 and now we have a great view about its columns and their relations, 
                 so we will just impor the data saved in the previous notebook and continue our journey. """)

        st.write("""We have already lowercased all of our data and implemented lemmatization and some other stuff, but just in case we will be doing some of that preprocessing again to make the model compatible to what we are pursuing.""""")
        

        st.subheader("An important part is to handel the imbalance of the data.")
        st.write("""Based on the train data size, we have decided not to oversample anything. Despite loosing some data, we will need less time adjusting our model and we can always go back to oversampling, but for now we will undersample all the classes to 50000 rows.""")


        # code 

        train = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Third Phaze\Second Problem\Data\train_finale.csv')
        test = pd.read_csv(r'D:\Sharif University of Tech\Data Sience Boot Camp\Project\Third Phaze\Second Problem\Data\test_finale.csv')
        train.drop(columns=["Unnamed: 0"], inplace=True)
        test.drop(columns=["Unnamed: 0"], inplace=True)
        train = train[["reviewText", "overall"]]
        test = test[["reviewText"]]
        undersampler = RandomUnderSampler(sampling_strategy={1: 50000, 2: 50000, 3: 50000, 4: 50000, 5: 50000}, random_state=78)
        X_resampled, y_resampled = undersampler.fit_resample(train.drop(columns=["overall"]), train['overall'])
        balanced_distribution = y_resampled.value_counts().sort_index()

        # end code 

        st.subheader("After balancing the rating distribution would be :")
        st.table(balanced_distribution)


        st.subheader("And the next procces would be to tokenize and lemmatize amd embedd the data.")


        st.header("Now that that's done lets implement some models :")
        st.write("For each model we will have a summary, its score and a panel for a potential user to interact with.")


        st.subheader("Choose a Model to Display:")
        model_choice = st.selectbox("Select a Model", ["SNN", "CNN", "RNN (LSTM)", "Test"])



        if model_choice == "SNN":
            st.subheader("The Summary for this model :")
            st.write("""
                Model: "sequential_2"
                _________________________________________________________________
                Layer (type)                Output Shape              Param #   
                _________________________________________________________________
                embedding (Embedding)       (None, 100, 100)          13026400  
                                                                                
                flatten (Flatten)           (None, 10000)             0         
                                                                                
                dense_6 (Dense)             (None, 6)                 60006     
                                                                                
                _________________________________________________________________
                Total params: 13086406 (49.92 MB)
                Trainable params: 60006 (234.40 KB)
                Non-trainable params: 13026400 (49.69 MB)
                _________________________________________________________________
                """)
            st.subheader("And its accuracy : 0.3649")


            st.write("As we can see this model is not very appealing.")



        if model_choice == "CNN":
            st.subheader("The Summary for this model :")
            st.write("""
                Model: "sequential_3"
                _________________________________________________________________
                Layer (type)                Output Shape              Param #   
                =================================================================
                embedding_1 (Embedding)     (None, 100, 100)          13026400  
                                                                                
                conv1d (Conv1D)             (None, 96, 128)           64128     
                                                                                
                global_max_pooling1d (Glob  (None, 128)               0         
                alMaxPooling1D)                                                 
                                                                                
                dense_7 (Dense)             (None, 6)                 774       
                                                                                
                =================================================================
                Total params: 13091302 (49.94 MB)
                Trainable params: 64902 (253.52 KB)
                Non-trainable params: 13026400 (49.69 MB)
                _________________________________________________________________               
                     """)
            st.subheader("And its accuracy : 0.4550")





        if model_choice == "RNN":
            st.subheader("The Summary for this model :")
            st.write("""
                Model: "sequential_4"
                _________________________________________________________________
                Layer (type)                Output Shape              Param #   
                =================================================================
                embedding_2 (Embedding)     (None, 100, 100)          13026400  
                                                                                
                lstm (LSTM)                 (None, 128)               117248    
                                                                                
                dense_8 (Dense)             (None, 6)                 774       
                                                                                
                =================================================================
                Total params: 13144422 (50.14 MB)
                Trainable params: 118022 (461.02 KB)
                Non-trainable params: 13026400 (49.69 MB)
                _________________________________________________________________
                """)
            st.subheader("And its accuracy : 0.5084")




        if model_choice == "Test":
            st.subheader("Now to test this hands on, we have this test box :")
            pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

            user_input = st.text_input("Enter your text here:")

            if user_input:

                result = pipe(user_input)[0]
                label = result['label']
                score = result['score'] * 100
                st.write(f"The example is {score:.2f}% {label}")






















