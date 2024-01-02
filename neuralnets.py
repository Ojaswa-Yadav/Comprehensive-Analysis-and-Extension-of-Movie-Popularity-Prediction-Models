import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# Load the data
#data = pd.read_excel("2014 and 2015 CSM dataset_labeld.xlsx")

train_conv = pd.read_excel('train_conv.xlsx')
test_conv = pd.read_excel('test_conv.xlsx')
train_social = pd.read_excel('train_social.xlsx')
test_social = pd.read_excel('test_social.xlsx')

def process_social_data(df):
    # Avoid division by zero: add a small number to dislikes
    df['Likes_Dislikes_Ratio'] = df['Likes'] / (df['Dislikes'] + 1e-8)
    df.drop(['Likes', 'Dislikes'], axis=1, inplace=True)
    return df

# Process the train and test social data
train_social = process_social_data(train_social)
test_social = process_social_data(test_social)

train_conv = train_conv.fillna(train_conv.median(numeric_only=True))
train_conv = train_conv.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'O' else x)
test_conv = test_conv.fillna(test_conv.median(numeric_only=True))
test_conv = test_conv.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'O' else x)
train_social = train_social.fillna(train_social.median())
test_social = test_social.fillna(test_social.median())

encoder = OneHotEncoder(sparse=False)
train_conv_encoded = encoder.fit_transform(train_conv[['Genre', 'Sequel']])
test_conv_encoded = encoder.transform(test_conv[['Genre', 'Sequel']])

encoded_columns = encoder.get_feature_names_out(['Genre', 'Sequel'])
train_conv = pd.concat([train_conv.drop(['Genre', 'Sequel'], axis=1), pd.DataFrame(train_conv_encoded, columns=encoded_columns)], axis=1)
test_conv = pd.concat([test_conv.drop(['Genre', 'Sequel'], axis=1), pd.DataFrame(test_conv_encoded, columns=encoded_columns)], axis=1)

# Combine conventional and social media features
train_combined = pd.concat([train_conv, train_social.reset_index(drop=True)], axis=1)
test_combined = pd.concat([test_conv, test_social.reset_index(drop=True)], axis=1)

y_train_gross = train_conv['GrossLabel']
y_test_gross = test_conv['GrossLabel']
y_train_rating = train_conv['RatingLabel']
y_test_rating = test_conv['RatingLabel']

# Drop labels from the combined datasets
train_conv = train_conv.drop(['GrossLabel', 'RatingLabel'], axis=1)
test_conv = test_conv.drop(['GrossLabel', 'RatingLabel'], axis=1)

train_social = train_social.drop(['GrossLabel', 'RatingLabel'], axis=1)
test_social = test_social.drop(['GrossLabel', 'RatingLabel'], axis=1)

train_combined = train_combined.drop(['GrossLabel', 'RatingLabel'], axis=1)
test_combined = test_combined.drop(['GrossLabel', 'RatingLabel'], axis=1)


gross_label_encoder = OneHotEncoder(sparse=False)
rating_label_encoder = OneHotEncoder(sparse=False)

# Encode GrossLabel
y_train_gross_encoded = gross_label_encoder.fit_transform(y_train_gross.values.reshape(-1, 1))
y_test_gross_encoded = gross_label_encoder.transform(y_test_gross.values.reshape(-1, 1))

# Encode RatingLabel
y_train_rating_encoded = rating_label_encoder.fit_transform(y_train_rating.values.reshape(-1, 1))
y_test_rating_encoded = rating_label_encoder.transform(y_test_rating.values.reshape(-1, 1))



scaler_conv = StandardScaler()
scaler_social = StandardScaler()
scaler_combined = StandardScaler()

# Fit and transform the training data
train_conv_scaled = scaler_conv.fit_transform(train_conv)
train_social_scaled = scaler_social.fit_transform(train_social)
train_combined_scaled = scaler_combined.fit_transform(train_combined)

# Transform the test data using the corresponding scaler
test_conv_scaled = scaler_conv.transform(test_conv)
test_social_scaled = scaler_social.transform(test_social)
test_combined_scaled = scaler_combined.transform(test_combined)

"""
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols] = data[num_cols].apply(lambda x: x.fillna(x.median()))

# For categorical features, fill with mode
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = data[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False)
encoded_cat = encoder.fit_transform(data[['Genre', 'Sequel']])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(['Genre', 'Sequel']))

# Combine encoded categorical features with the rest of the dataset
data = data.drop(['Genre', 'Sequel'], axis=1)
data = pd.concat([data, encoded_cat_df], axis=1)

# Split the data into features and labels
X_conv = data[['Budget', 'Screens'] + list(encoded_cat_df.columns)]  # Conventional features
X_social = data[['Sentiment', 'Views', 'Likes', 'Dislikes', 'Comments', 'Aggregate Followers']]  # Social media features
X_combined = pd.concat([X_conv, X_social], axis=1)

y_gross = data['GrossLabel']
y_rating = data['RatingLabel']

# Preprocessing
scaler = StandardScaler()
X_conv_scaled = scaler.fit_transform(X_conv)
X_social_scaled = scaler.fit_transform(X_social)
X_combined_scaled = scaler.fit_transform(X_combined)

# Convert labels to one-hot encoded format
y_gross_encoded = encoder.fit_transform(y_gross.values.reshape(-1, 1))
y_rating_encoded = encoder.fit_transform(y_rating.values.reshape(-1, 1))

# Split data into training and test sets
X_train_conv, X_test_conv, y_train_gross, y_test_gross = train_test_split(X_conv_scaled, y_gross_encoded, test_size=0.2, random_state=100)
X_train_social, X_test_social, y_train_gross_social, y_test_gross_social = train_test_split(X_social_scaled, y_gross_encoded, test_size=0.2, random_state=100)
X_train_combined, X_test_combined, y_train_gross_combined, y_test_gross_combined = train_test_split(X_combined, y_gross_encoded, test_size=0.2, random_state=100)

X_train_conv_rating, X_test_conv_rating, y_train_rating_conv, y_test_rating_conv = train_test_split(X_conv_scaled, y_rating_encoded, test_size=0.2, random_state=100)
X_train_social_rating, X_test_social_rating, y_train_rating_social, y_test_rating_social = train_test_split(X_social_scaled, y_rating_encoded, test_size=0.2, random_state=100)
X_train_combined_rating, X_test_combined_rating, y_train_rating_combined, y_test_rating_combined = train_test_split(X_combined_scaled, y_rating_encoded, test_size=0.2, random_state=100)
"""
# Define the neural network model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(X_train, y_train, X_test, y_test, input_shape, num_classes):
    model = build_model(input_shape, num_classes)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return model, history, accuracy

n_runs = 1  # Number of times to train each model
best_models = {}  # Dictionary to store the best models

def train_and_select_best_model(train_func, X_train, y_train, X_test, y_test, input_shape, num_classes, model_name):
    best_accuracy = 0
    best_model = None
    best_loss = None

    for i in range(n_runs):
        print(f"Training {model_name}, Run {i + 1}/{n_runs}")
        model, history, accuracy = train_func(X_train, y_train, X_test, y_test, input_shape, num_classes)

        # Evaluate and check if this model is the best so far
        loss = history.history['val_loss'][-1]  # Last validation loss
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_history = history
            best_model = model

    print(f"Best model for {model_name} achieved an accuracy of: {best_accuracy}")
    return best_model, best_history, best_accuracy


model_gross_conv, loss_gross_conv, accuracy_gross_conv = train_and_select_best_model(
    train_and_evaluate_model, train_conv_scaled, y_train_gross_encoded,
    test_conv_scaled, y_test_gross_encoded, train_conv_scaled.shape[1], y_train_gross_encoded.shape[1],
    'GrossLabel - Conventional')

model_gross_social, loss_gross_social, accuracy_gross_social  = train_and_select_best_model(
    train_and_evaluate_model, train_social_scaled, y_train_gross_encoded,
    test_social_scaled, y_test_gross_encoded, train_social_scaled.shape[1], y_train_gross_encoded.shape[1],
    'GrossLabel - Social')

# Train and select the best models for GrossLabel - Combined Features
model_gross_combined, loss_gross_combined, accuracy_gross_combined = train_and_select_best_model(
    train_and_evaluate_model, train_combined_scaled, y_train_gross_encoded,
    test_combined_scaled, y_test_gross_encoded, train_combined_scaled.shape[1], y_train_gross_encoded.shape[1],
    'GrossLabel - Combined')

# Train and select the best models for RatingLabel - Conventional Features
model_rating_conv, loss_rating_conv, accuracy_rating_conv = train_and_select_best_model(
    train_and_evaluate_model, train_conv_scaled, y_train_rating_encoded,
    test_conv_scaled, y_test_rating_encoded, train_conv_scaled.shape[1], y_train_rating_encoded.shape[1],
    'RatingLabel - Conventional')

# Train and select the best models for RatingLabel - Social Media Features
model_rating_social, loss_rating_social, accuracy_rating_social= train_and_select_best_model(
    train_and_evaluate_model, train_social_scaled, y_train_rating_encoded,
    test_social_scaled, y_test_rating_encoded, train_social_scaled.shape[1], y_train_rating_encoded.shape[1],
    'RatingLabel - Social')

# Train and select the best models for RatingLabel - Combined Features
model_rating_combined, loss_rating_combined, accuracy_rating_combined = train_and_select_best_model(
    train_and_evaluate_model, train_combined_scaled, y_train_rating_encoded,
    test_combined_scaled, y_test_rating_encoded, train_combined_scaled.shape[1], y_train_rating_encoded.shape[1],
    'RatingLabel - Combined')


# Print out the accuracies
print("Accuracy for GrossLabel Prediction:")
print(f" - Conventional Features: {accuracy_gross_conv}")
print(f" - Social Media Features: {accuracy_gross_social}")
print(f" - Combined Features: {accuracy_gross_combined}")

print("\nAccuracy for RatingLabel Prediction:")
print(f" - Conventional Features: {accuracy_rating_conv}")
print(f" - Social Media Features: {accuracy_rating_social}")
print(f" - Combined Features: {accuracy_rating_combined}")



def plot_training_history(history, title):
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

# Plot training history for each model
plot_training_history(loss_gross_conv, "GrossLabel - Conventional Features")
plot_training_history(loss_gross_social, "GrossLabel - Social Features")
plot_training_history(loss_gross_combined, "GrossLabel - Combined Features")
plot_training_history(loss_rating_conv, "RatingLabel - Conventional Features")
plot_training_history(loss_rating_social, "RatingLabel - Social Features")
plot_training_history(loss_rating_combined, "RatingLabel - Combined Features")

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Print the confusion matrix values on the plot
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Get predictions from models
def get_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred

# Generate predictions for all models
y_pred_gross_conv = get_predictions(model_gross_conv, test_conv_scaled)
y_pred_gross_social = get_predictions(model_gross_social, test_social_scaled)
y_pred_gross_combined = get_predictions(model_gross_combined, test_combined_scaled)

y_pred_rating_conv = get_predictions(model_rating_conv, test_conv_scaled)
y_pred_rating_social = get_predictions(model_rating_social, test_social_scaled)
y_pred_rating_combined = get_predictions(model_rating_combined, test_combined_scaled)

y_test_gross = np.argmax(y_test_gross_encoded, axis=1)
y_test_rating = np.argmax(y_test_rating_encoded, axis=1)

# Compute confusion matrices
cm_gross_conv = confusion_matrix(y_test_gross, y_pred_gross_conv)
cm_gross_social = confusion_matrix(y_test_gross, y_pred_gross_social)
cm_gross_combined = confusion_matrix(y_test_gross, y_pred_gross_combined)

cm_rating_conv = confusion_matrix(y_test_rating, y_pred_rating_conv)
cm_rating_social = confusion_matrix(y_test_rating, y_pred_rating_social)
cm_rating_combined = confusion_matrix(y_test_rating, y_pred_rating_combined)

gross_classes = gross_label_encoder.categories_[0]
rating_classes = rating_label_encoder.categories_[0]

# Plot confusion matrices
plt.figure(figsize=(18, 10))

# GrossLabel confusion matrices
plt.subplot(2, 3, 1)
plot_confusion_matrix(cm_gross_conv, classes=gross_classes, title='GrossLabel - Conventional Features')

plt.subplot(2, 3, 2)
plot_confusion_matrix(cm_gross_social, classes=gross_classes, title='GrossLabel - Social Media Features')

plt.subplot(2, 3, 3)
plot_confusion_matrix(cm_gross_combined, classes=gross_classes, title='GrossLabel - Combined Features')

# RatingLabel confusion matrices
plt.subplot(2, 3, 4)
plot_confusion_matrix(cm_rating_conv, classes=rating_classes, title='RatingLabel - Conventional Features')

plt.subplot(2, 3, 5)
plot_confusion_matrix(cm_rating_social, classes=rating_classes, title='RatingLabel - Social Media Features')

plt.subplot(2, 3, 6)
plot_confusion_matrix(cm_rating_combined, classes=rating_classes, title='RatingLabel - Combined Features')

plt.show()

# Initialize an empty list to store the metrics
model_metrics = []

# Function to calculate metrics and append to the list
def calculate_metrics(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    model_metrics.append({
        'Model': model_name,
        'Accuracy': report['accuracy'],
        'Precision': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall'],
        'F1': report['macro avg']['f1-score']
    })

# Calculate metrics for each model
calculate_metrics(y_test_gross, y_pred_gross_conv, 'GrossLabel - Conventional')
calculate_metrics(y_test_gross, y_pred_gross_social, 'GrossLabel - Social')
calculate_metrics(y_test_gross, y_pred_gross_combined, 'GrossLabel - Combined')
calculate_metrics(y_test_rating, y_pred_rating_conv, 'RatingLabel - Conventional')
calculate_metrics(y_test_rating, y_pred_rating_social, 'RatingLabel - Social')
calculate_metrics(y_test_rating, y_pred_rating_combined, 'RatingLabel - Combined')

# Convert the list to a DataFrame
metrics_df = pd.DataFrame(model_metrics)

# Save the DataFrame to a CSV file
#metrics_df.to_csv('model_nn_metrics.csv', index=False)

print("CSV file with model metrics has been created successfully.")

# Sample data: replace these with your actual accuracy scores
accuracy_scores = {
    'GrossLabel': {
        'Social': metrics_df['Accuracy'][1],
        'Conventional': metrics_df['Accuracy'][0],
        'Combined': metrics_df['Accuracy'][2]
    },
    'RatingLabel': {
        'Social': metrics_df['Accuracy'][3],
        'Conventional': metrics_df['Accuracy'][4],
        'Combined': metrics_df['Accuracy'][5]
    }
}

# Define the label locations and width of the bars
labels = ['GrossLabel', 'RatingLabel']
social_scores = [accuracy_scores['GrossLabel']['Social'], accuracy_scores['RatingLabel']['Social']]
conventional_scores = [accuracy_scores['GrossLabel']['Conventional'], accuracy_scores['RatingLabel']['Conventional']]
combined_scores = [accuracy_scores['GrossLabel']['Combined'], accuracy_scores['RatingLabel']['Combined']]
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()

# Plot the bars
rects1 = ax.bar(x - width, social_scores, width, label='Social', color='blue')
rects2 = ax.bar(x, conventional_scores, width, label='Conventional', color='red')
rects3 = ax.bar(x + width, combined_scores, width, label='Combined', color='green')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
