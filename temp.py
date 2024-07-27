import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QLabel, QLineEdit, QPushButton, QListWidget, QMessageBox, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QFont, QColor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam


class BOOKS(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BOOKS GUI')
        self.setGeometry(200, 200, 300, 200)
        self.setFixedSize(700, 250)
        color = QColor.fromHsv(190, 255, 255)  # Hue: 190, Saturation: 255, Value: 255
        self.setStyleSheet(f"background-color: {color.name()};")

        self.username_label = QLabel('Username:', self)
        self.username_label.move(220, 50)
        self.username_text = QLineEdit(self)
        self.username_text.move(320, 50)

        self.password_label = QLabel('Password:', self)
        self.password_label.move(220, 80)
        self.password_text = QLineEdit(self)
        self.password_text.setEchoMode(QLineEdit.Password)
        self.password_text.move(320, 80)

        self.login_button = QPushButton('Login', self)
        self.login_button.clicked.connect(self.login)
        self.login_button.move(260, 135)

        self.register_button = QPushButton('Register', self)
        self.register_button.clicked.connect(self.register)
        self.register_button.move(370, 135)

        self.logout_button = QPushButton('Logout', self)
        self.logout_button.clicked.connect(self.logout)
        self.logout_button.move(610, 220)
        self.logout_button.hide()

        self.login_message1 = QLabel('Login now and let the excitement begin!!', self)
        self.login_message1.move(250, 180)
        self.login_message1.setFont(QFont("Arial", 10))  # Set the font to smaller size

        self.login_message2 = QLabel('Don\'t have an account? Just register the desired username and password and you\'re good to go!', self)
        self.login_message2.setFont(QFont("Arial", 10))
        self.login_message2.move(100, 200)

        self.username = None

        # Add the "BOOKS" button
        self.books_button = QPushButton('BOOKS', self)
        self.books_button.move(50, 135)
        self.books_button.clicked.connect(self.show_books)
        self.books_button.hide()  # Initially hide the button, show it after login

        # Add the "Interested Books" button
        self.interested_books_button = QPushButton('Interested Books', self)
        self.interested_books_button.move(170, 135)
        self.interested_books_button.clicked.connect(self.show_interested_books)
        self.interested_books_button.hide()  # Initially hide the button, show it after login

        # Add the "Recommended Books" button
        self.recommended_books_button = QPushButton('Recommended Books', self)
        self.recommended_books_button.move(390, 135)
        self.recommended_books_button.clicked.connect(self.show_recommended_books)
        self.recommended_books_button.hide()  # Initially hide the button, show it after login

        # List widget to show books
        self.books_list = QListWidget(self)
        self.books_list.setGeometry(50, 50, 600, 150)
        self.books_list.itemClicked.connect(self.book_details)
        self.books_list.hide()

        # Add the "Back" button
        self.back_button = QPushButton('Back', self)
        self.back_button.move(50, 210)
        self.back_button.clicked.connect(self.go_back)
        self.back_button.hide()  # Initially hide the button

        self.update_login_page_messages()

    def update_login_page_messages(self):
        if self.username is None:
            self.login_message1.show()
            self.login_message2.show()
        else:
            self.login_message1.hide()
            self.login_message2.hide()

    def login(self):
        username = self.username_text.text()
        password = self.password_text.text()

        if username == '' and password == '':
            QMessageBox.warning(self, 'Login', 'Please enter both username and password!')
            return
        elif username == '':
            QMessageBox.warning(self, 'Login', 'Please enter the username!')
            return
        elif password == '':
            QMessageBox.warning(self, 'Login', 'Please enter the password!')
            return

        if self.authenticate(username, password):
            self.username = username
            self.show_BOOK_selection()
            self.update_login_page_messages()
        else:
            QMessageBox.warning(self, 'Login', 'Invalid username or password!')

    def register(self):
        username = self.username_text.text()
        password = self.password_text.text()

        if username == '':
            QMessageBox.warning(self, 'Registration', 'Please enter a username!')
            return
        elif password == '':
            QMessageBox.warning(self, 'Registration', 'Please enter a password!')
            return

        if self.username_exists(username):
            QMessageBox.warning(self, 'Registration', 'Username already exists!')
        else:
            self.create_account(username, password)
            QMessageBox.information(self, 'Registration', 'Registration successful!\nCongratulations for 1000 tokens!')

    def create_account(self, username, password):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        login_file = os.path.join(script_dir, 'login.txt')

        with open(login_file, 'a') as file:
            file.write(f'{username}:{password}\n')

        tokens_file = os.path.join(script_dir, 'tokens.txt')

        with open(tokens_file, 'a') as file:
            file.write(f'{username}:0\n')

    def username_exists(self, username):
        with open('login.txt', 'r') as file:
            login = file.readlines()
            for account in login:
                if username == account.split(':')[0]:
                    return True
        return False

    def authenticate(self, username, password):
        with open('login.txt', 'r') as file:
            login = file.readlines()
            for account in login:
                if f'{username}:{password}\n' == account:
                    return True
        return False

    def show_BOOK_selection(self):
        self.username_label.hide()
        self.username_text.hide()
        self.password_label.hide()
        self.password_text.hide()
        self.login_button.hide()
        self.register_button.hide()
        self.logout_button.show()
        self.books_button.show()
        self.interested_books_button.show()
        self.recommended_books_button.show()
        welcome_message = f'Welcome, {self.username}! Take your pick and let the fun begin!'
        self.welcome_label = QLabel(welcome_message, self)
        self.welcome_label.move(150, 20)
        self.welcome_label.show()

    def show_books(self):
        self.books_list.clear()
        self.books_list.show()
        self.back_button.show()  # Show the back button

        # Load the book dataset
        dataset_path = 'C://Users//Dell//Desktop//BooksDatasetClean.csv'
        books_df = pd.read_csv(dataset_path)

        # Store the dataset for later use
        self.books_df = books_df

        # Add books to the list
        for index, book in books_df.iterrows():
            self.books_list.addItem(book['Title'])

    def book_details(self, item):
        book_title = item.text()
        book = self.books_df[self.books_df['Title'] == book_title].iloc[0]

        interested_books = self.get_interested_books()
        interested = book_title in interested_books

        dialog = QDialog(self)
        dialog.setWindowTitle('Book Details')
        dialog.setGeometry(300, 300, 400, 400)
        layout = QVBoxLayout()

        for col in book.index:
            if col == 'Description':
                description = book[col]
                if pd.isna(description):
                    description = 'No description available.'
                description_label = QLabel(f"{col}:")
                layout.addWidget(description_label)
                description_text = QTextEdit(description)
                description_text.setReadOnly(True)
                layout.addWidget(description_text)
            else:
                label = QLabel(f"{col}: {book[col]}")
                layout.addWidget(label)

        button_text = 'Uninterested' if interested else 'Interested'
        interested_button = QPushButton(button_text, dialog)
        interested_button.clicked.connect(lambda: self.toggle_interest(book_title, interested_button, dialog))
        layout.addWidget(interested_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def get_interested_books(self):
        try:
            with open('interested_books.txt', 'r') as file:
                books = [line.strip() for line in file.readlines()]
            return books
        except FileNotFoundError:
            return []

    def toggle_interest(self, book_title, button, dialog=None):
        interested_books = self.get_interested_books()
        if book_title in interested_books:
            interested_books.remove(book_title)
            with open('interested_books.txt', 'w') as file:
                for book in interested_books:
                    file.write(book + '\n')
            button.setText('Interested')
            QMessageBox.information(self, 'Uninterested', f'You have removed "{book_title}" from interested books.')
        else:
            with open('interested_books.txt', 'a') as file:
                file.write(book_title + '\n')
            button.setText('Uninterested')
            QMessageBox.information(self, 'Interested', f'You have marked "{book_title}" as interested.')

        # Refresh the interested books list if dialog is for interested books
        if dialog:
            self.refresh_interested_books_list(dialog)

    def refresh_interested_books_list(self, dialog):
        dialog.close()
        self.show_interested_books()

    def show_interested_books(self):
        interested_books_dialog = QDialog(self)
        interested_books_dialog.setWindowTitle('Interested Books')
        interested_books_dialog.setGeometry(300, 300, 400, 400)
        layout = QVBoxLayout()

        interested_books_list = QListWidget(interested_books_dialog)
        try:
            with open('interested_books.txt', 'r') as file:
                books = file.readlines()
                for book in books:
                    interested_books_list.addItem(book.strip())
            interested_books_list.itemClicked.connect(self.book_details)  # Connect to book_details method
        except FileNotFoundError:
            QMessageBox.warning(self, 'Interested Books', 'No books have been marked as interested yet.')

        layout.addWidget(interested_books_list)
        interested_books_dialog.setLayout(layout)
        interested_books_dialog.exec_()

    def show_recommended_books(self):
        if not hasattr(self, 'books_df'):
            QMessageBox.warning(self, 'Error', 'Books data not loaded. Please click on the "BOOKS" button first.')
            return

        interested_books = self.get_interested_books()
        if not interested_books:
            QMessageBox.warning(self, 'Recommended Books', 'No interested books found for recommendations.')
            return

        # Load the saved model
        model = load_model('book_recommendation_model.h5')

        categories = set()
        for book_title in interested_books:
            book = self.books_df[self.books_df['Title'] == book_title].iloc[0]
            categories.add(book['Category'])

        recommended_books = self.books_df[self.books_df['Category'].isin(categories) & ~self.books_df['Title'].isin(interested_books)]
        
        recommended_books_dialog = QDialog(self)
        recommended_books_dialog.setWindowTitle('Recommended Books')
        recommended_books_dialog.setGeometry(300, 300, 400, 400)
        layout = QVBoxLayout()

        recommended_books_list = QListWidget(recommended_books_dialog)
        for index, book in recommended_books.iterrows():
            X_pred = self.prepare_input_for_model(book)
            category_index = np.argmax(model.predict(X_pred), axis=-1)[0]
            recommended_books_list.addItem(f"{book['Title']} - Predicted Category: {category_index}")
        recommended_books_list.itemClicked.connect(lambda item: self.book_details(item))
        
        layout.addWidget(recommended_books_list)
        recommended_books_dialog.setLayout(layout)
        recommended_books_dialog.exec_()

    def prepare_input_for_model(self, book):
        features = ['Title', 'Authors', 'Category', 'Publisher', 'Price Starting With ($)', 'Publish Date (Month)', 'Publish Date (Year)']
        X_pred = {
            'Title': np.array([book['Title']]),
            'Authors': np.array([book['Authors']]),
            'Category': np.array([book['Category']]),
            'Publisher': np.array([book['Publisher']]),
            'Price Starting With ($)': np.array([book['Price Starting With ($)']]),
            'Publish Date (Month)': np.array([book['Publish Date (Month)']]),
            'Publish Date (Year)': np.array([book['Publish Date (Year)']])
        }
        return X_pred

    def go_back(self):
        self.books_list.hide()
        self.back_button.hide()
        self.username_label.hide()
        self.username_text.hide()
        self.password_label.hide()
        self.password_text.hide()
        self.login_button.hide()
        self.register_button.hide()
        self.logout_button.show()
        self.books_button.show()
        self.interested_books_button.show()
        self.recommended_books_button.show()
        self.welcome_label.show()

    def logout(self):
        self.username = None
        self.username_label.show()
        self.username_text.show()
        self.username_text.setText('')
        self.password_label.show()
        self.password_text.show()
        self.password_text.setText('')
        self.login_button.show()
        self.register_button.show()
        self.logout_button.hide()
        self.books_button.hide()
        self.interested_books_button.hide()
        self.recommended_books_button.hide()
        self.update_login_page_messages()
        self.welcome_label.hide()
        self.books_list.hide()
        self.back_button.hide()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont('Verdana', 12)
    app.setFont(font)
    BOOK_app = BOOKS()

    print("Training model...")
    # Load the dataset
    file_path = 'C://Users//Dell//Desktop//BooksDatasetClean.csv'
    data = pd.read_csv(file_path)

    # Fill NaN values with empty strings
    data.fillna('', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in ['Title', 'Authors', 'Category', 'Publisher']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Convert month names to numbers
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    data['Publish Date (Month)'] = data['Publish Date (Month)'].map(month_map)

    # Separate features and target
    features = ['Title', 'Authors', 'Category', 'Publisher', 'Price Starting With ($)', 'Publish Date (Month)', 'Publish Date (Year)']
    X = data[features]
    y = data['Category']

    # Normalize numerical features
    scaler = StandardScaler()
    X[['Price Starting With ($)', 'Publish Date (Month)', 'Publish Date (Year)']] = scaler.fit_transform(X[['Price Starting With ($)', 'Publish Date (Month)', 'Publish Date (Year)']])

    # Define input layers
    input_layers = []
    embedding_layers = []

    # Create embedding layers for categorical features
    for column in ['Title', 'Authors', 'Category', 'Publisher']:
        input_layer = Input(shape=(1,), name=column)
        input_layers.append(input_layer)
        vocab_size = data[column].nunique() + 1
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=50)(input_layer)
        embedding_layer = Flatten()(embedding_layer)
        embedding_layers.append(embedding_layer)

    # Numerical features
    numerical_features = ['Price Starting With ($)', 'Publish Date (Month)', 'Publish Date (Year)']
    for column in numerical_features:
        input_layer = Input(shape=(1,), name=column)
        input_layers.append(input_layer)
        embedding_layers.append(input_layer)

    # Concatenate all embedding layers
    concat_layer = Concatenate()(embedding_layers)

    # Add dense layers
    dense_layer = Dense(128, activation='relu')(concat_layer)
    dense_layer = Dropout(0.2)(dense_layer)
    dense_layer = Dense(64, activation='relu')(dense_layer)
    dense_layer = Dropout(0.2)(dense_layer)
    output_layer = Dense(data['Category'].nunique(), activation='softmax')(dense_layer)

    # Build the model
    model = Model(inputs=input_layers, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Prepare the inputs for the model
    X_dict = {col: X[col].values for col in features}
    
    history = model.fit(X_dict, y, epochs=2, batch_size=32, validation_split=0.2, verbose=1)

    # Display training history
    print("Model training completed.")
    print("Training History:")
    print(history.history)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_dict, y)
    print(f'Training Loss: {loss}')
    print(f'Training Accuracy: {accuracy}')

    BOOK_app.show()  # Show the GUI
    sys.exit(app.exec_())  # Execute the application event loop


