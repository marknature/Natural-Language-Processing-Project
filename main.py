import pandas as pd
import re
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load datasets
true1 = pd.read_csv("dataset/True.csv")
true2 = pd.read_csv("dataset/second_set/real.csv")
false1 = pd.read_csv("dataset/Fake.csv")
false2 = pd.read_csv("dataset/second_set/fake.csv")
both_news = pd.read_csv("dataset/WELFake_Dataset.csv/WELFake_Dataset.csv")

# Combine title and text columns
true1["text"] = true1["title"] + " " + true1["text"]
false1["text"] = false1["title"] + " " + false1["text"]
both_news["text"] = both_news["title"] + " " + both_news["text"]

# Assign categories for classification
true1["category"] = 1
true2["category"] = 1
false1["category"] = 0
false2["category"] = 0
both_news.rename(columns={"Label": "category"}, inplace=True)  # Rename label column for new dataset

# Concatenate true and fake datasets
true = pd.concat([true1, true2]).reset_index(drop=True)
false = pd.concat([false1, false2]).reset_index(drop=True)
# check
# print(true.head())
# print(false.tail())
# print(true.shape)
# print(false.shape)
# print(both_news.shape)


# Handle missing values
def handle_missing_values(dframe):
    return dframe.dropna()


# Apply missing values handling
true = handle_missing_values(true)
false = handle_missing_values(false)
both_news = handle_missing_values(both_news)

# Combine true and fake news datasets
news = pd.concat([true, false, both_news]).reset_index(drop=True)
# print(news.describe())
# print(news.columns)
# print(news.shape)
# print(news.info())


# Preprocess the text data
def preprocess_text(txt):
    # Check if the value is NaN
    if pd.isna(txt):
        return ""
    # Convert into lowercase
    txt = txt.lower()
    # remove URLs
    txt = re.sub(r'https?://\S+|www\.\S+', '', txt)
    # remove HTML tags
    txt = re.sub(r'<.*?>', '', txt)
    # remove punctuation
    txt = re.sub(r'[^\w\s]', '', txt)
    # remove digits
    txt = re.sub(r'\d', '', txt)
    # remove newline characters
    txt = re.sub(r'\n', ' ', txt)
    return txt


# checking if there is empty string in TEXT column
blanks = []
# index,label and review of the doc
for index, text in news["text"].items():  # it will iter through index,label and review
    if text.isspace():  # if there is a space
        blanks.append(index)  # it will be noted down in empty list
# print(len(blanks))

# instead of dropping these values we are going to merge title with text
news["text"] = news["title"] + news["text"]
# we only need two columns rest can be-ignore
news = news[["text", "category"]]
# Shuffle the dataset
news = news.sample(frac=1).reset_index(drop=True)

# creating instance
lemma = WordNetLemmatizer()
# creating list of stopwords containing stopwords from spacy and nltk
# stopwords of NLTK
Stopwords = stopwords.words('english')
# print(len(Stopwords))


def clean_text(text):
    string = ""
    # lower casing
    text = text.lower()
    # simplifying text
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    # removing any special character
    text = re.sub(r"[-()\"#!@$%^&*{}?.,:]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    for word in text.split():
        if word not in Stopwords:
            string += lemma.lemmatize(word) + " "
    return string


# apply both our functions for cleaning
news['text'] = news['text'].apply(preprocess_text)
news['text'] = news['text'].apply(clean_text)

# Remove rows with blank text
news = news[news['text'].str.strip() != ""]

# Feature-Extraction & Model building
x = news['text']
y = news['category']

# Check for missing values in the target variable
# print(news.isna().sum()*100/len(news))
if y.isnull().values.any():
    # print("Target variable contains missing values. Handling missing values...")
    news = news.dropna(subset=['category'])
    x = news['text']
    y = news['category']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create and train the SVM model using a pipeline
text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])
text_clf.fit(x_train, y_train)

# Make predictions and evaluate the SVM model
# this pipe-line will take the text and vectorise it , and then TF-IDF, then fitting the model
predictions = text_clf.predict(x_test)
print("SVM Model")
# confusion matrix
print("Confusion matrix: ")
print(metrics.confusion_matrix(y_test, predictions))
# overall accuracy
print("Overall accuracy: ", metrics.accuracy_score(y_test, predictions))
print(metrics.classification_report(y_test, predictions))

# Vectorize the text data for other models
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Define and train other classification models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier()
}

for model_name, model in models.items():
    model.fit(xv_train, y_train)
    predictions = model.predict(xv_test)
    print(model_name)
    print("Accuracy:", model.score(xv_test, y_test))
    print(classification_report(y_test, predictions))


# Manual testing function
def output_label(label):
    return "Genuine News" if label == 1 else "Fake News"


def manual_testing(text):
    preprocessed_text = preprocess_text(text)
    new_xv_test = vectorization.transform([preprocessed_text])

    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(new_xv_test)
        predictions[model_name] = output_label(prediction[0])

    return predictions


# Example of manual testing
input_text = input("Enter your Article: ")
news_article = manual_testing(input_text)
for model_name, prediction in news_article.items():
    print(f"{model_name} Prediction: {prediction}")
