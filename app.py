#importing required packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from URLFeatureExtraction import featureExtraction, feature_names 
import warnings

# Disable warnings
warnings.filterwarnings("ignore")

st.title("Phishing Website Detection")
st.write(f"Identify and prevent phishing attacks with real-time URL analysis.")

# Load dataset
data = pd.read_csv('./5.urldata.csv')

# Feature Extraction
y = data['Label']
X = data.drop( ['Domain','Label','Web_Traffic','Right_Click','https_Domain','Redirection','Mouse_Over','DNS_Record'] ,axis=1)
x = data['Domain']
     

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 12)
from xgboost import XGBClassifier

# instantiate the model
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
xgb.fit(X_train, y_train)
     
#predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

#computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train,y_train_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)


tab1, tab2, tab3, tab4 = st.tabs(["Check website","Information", "DataSet", "Model Accuracy"]) 

with tab1:
	url_input = st.text_input("Enter URL")

	if st.button("Check"):
		if url_input:
			user_input_features = featureExtraction(url_input)
			prediction = xgb.predict(user_input_features)
			if prediction[0] == 1:
				st.error("This URL is identified as a phishing site!")
			else:
				st.success("This URL is safe.")
		else:
			st.warning("Please enter a URL before checking!")
with tab2:
	st.write('''
			A phishing website detection application aims to identify malicious websites that attempt to deceive users into revealing sensitive information such as usernames, passwords, and credit card details. These applications leverage various techniques and machine learning models to analyze and determine whether a given URL is legitimate or potentially harmful.
	 ### Key Characteristics of Phishing Websites

1.  **URL Deception**:

    -   **Similar Domain Names**: Phishing websites often use domain names that are very similar to legitimate sites, with slight variations (e.g., using numbers instead of letters, additional words, or different top-level domains like `.com` vs. `.net`).
    -   **Use of Subdomains**: They might use subdomains to give the appearance of legitimacy (e.g., `login.bank.example.com` instead of `bank.example.com`).
2.  **Visual Imitation**:

    -   **Design and Layout**: Phishing sites often copy the design, logos, and overall layout of the legitimate websites they are impersonating.
    -   **Fake Forms**: They include forms that ask for personal information such as login credentials, payment details, or social security numbers.
3.  **Insecure Connections**:

    -   **Lack of HTTPS**: While not always a sign of phishing, many phishing websites lack HTTPS encryption, making the connection insecure.
4.  **Suspicious Content**:

    -   **Urgency and Threats**: Phishing websites often contain urgent messages or threats (e.g., "Your account will be suspended unless you verify your information immediately").
    -   **Unusual Requests**: They ask for information that legitimate websites typically wouldn't request, such as PIN numbers or full social security numbers via web forms.
5.  **Technical Indicators**:

    -   **IP Address in URL**: Legitimate websites usually have domain names, not IP addresses (e.g., `http://192.168.1.1/login`).
    -   **Poor Grammar and Spelling**: Many phishing websites have spelling errors and poor grammar.

### Techniques Used by Phishing Websites

1.  **Email Phishing**: Sending fake emails that appear to come from trusted organizations, prompting users to click on a link that leads to a phishing website.
2.  **Spear Phishing**: Targeting specific individuals or organizations with personalized messages to increase the likelihood of deceiving them.
3.  **Clone Phishing**: Creating a nearly identical copy of a legitimate email that contains a link to the phishing website.
4.  **Website Redirection**: Redirecting traffic from a legitimate website to a phishing site through malicious ads, compromised web pages, or shortened URLs.

### Detection Techniques

To determine if a website is phishing, various detection techniques can be employed:

1.  **URL Analysis**:

    -   Check for suspicious patterns in the URL (e.g., excessive use of special characters, IP addresses, or unusual domains).
2.  **Content Analysis**:

    -   Analyze the website's content for phishing indicators such as urgent messages, requests for sensitive information, and inconsistencies in design.
3.  **Machine Learning Models**:

    -   Train models using features extracted from both phishing and legitimate websites to predict the likelihood of a site being phishing.
4.  **Database Comparison**:

    -   Compare the URL against known phishing databases like PhishTank and OpenPhish.
		  
		  ''')	
with tab3:
	st.write('''##### The Dataset used for training the model:''')
	st.dataframe(data.head())
	st.write('''##### The List of features include:''')

	n = len(feature_names)//2
	names_column1 = feature_names[:n+1]
	names_column2 = feature_names[n+1:]

	col1, col2 = st.columns(2)
	with col1:
		for name in names_column1:
			st.markdown(f"- {name}")
	with col2:
		for name in names_column2:
			st.markdown(f"- {name}")

	#Correlation heatmap
	fig1, ax = plt.subplots()
	temp = data.drop( ['Domain'] ,axis=1)
	sns.heatmap(temp.corr())
	st.write("\n##### Heatmap:")
	st.pyplot(fig1)

	#checking the feature improtance in the model
	st.write("\n##### Feature Importance:")
	fig2, ax = plt.subplots()
	n_features = X_train.shape[1]
	plt.barh(range(n_features), xgb.feature_importances_, align='center',color = "#FF4B4B")
	plt.yticks(np.arange(n_features), X_train.columns)
	plt.xlabel("Feature importance")
	plt.ylabel("Feature")
	st.pyplot(fig2) 
with tab4:
	st.write('''##### Train accuracy:''',acc_train_xgb)
	st.write('''##### Test accuaracy:''',acc_test_xgb)

	cm = confusion_matrix(y_test,y_test_xgb)
	cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

	# Create a heatmap for the confusion matrix
	fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size
	sns.heatmap(cm_df, annot=True, fmt='d', cmap='OrRd', ax=ax)  # Use a different color palette
	ax.set_xlabel('Predicted Labels')
	ax.set_ylabel('True Labels')
	st.write('''##### Confusion matrix:''')
	# Display the confusion matrix in Streamlit
	st.pyplot(fig)
	



        
