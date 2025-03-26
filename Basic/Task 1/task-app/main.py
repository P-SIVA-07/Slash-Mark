import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    tasks = pd.DataFrame(columns=['description', 'priority'])

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Ensure there's data before training
if not tasks.empty:
    vectorizer = CountVectorizer()
    clf = MultinomialNB()
    model = make_pipeline(vectorizer, clf)
    model.fit(tasks['description'], tasks['priority'])
else:
    model = None  # No model training if there's no data

# Function to add a task to the list
def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()
    print(f"Task '{description}' added successfully!")

# Function to remove a task by description
def remove_task(description):
    global tasks
    if description in tasks['description'].values:
        tasks = tasks[tasks['description'] != description].copy()
        save_tasks()
        print(f"Task '{description}' removed.")
    else:
        print("Task not found.")

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Function to recommend a task based on machine learning
def recommend_task():
    if tasks.empty:
        print("No tasks available for recommendation.")
        return
    
    high_priority_tasks = tasks[tasks['priority'].str.lower() == 'high']
    if high_priority_tasks.empty:
        print("No high-priority tasks found.")
        return
    
    random_task = random.choice(high_priority_tasks['description'].tolist())
    print(f"Recommended Task: {random_task}")

# Main menu loop
def main():
    while True:
        print("\n1. Add Task\n2. Remove Task\n3. List Tasks\n4. Recommend Task\n5. Exit")
        choice = input("Select an option: ")
        
        if choice == '1':
            desc = input("Enter task description: ")
            priority = input("Enter priority (low/medium/high): ").strip().lower()
            add_task(desc, priority)
        elif choice == '2':
            desc = input("Enter task description to remove: ")
            remove_task(desc)
        elif choice == '3':
            list_tasks()
        elif choice == '4':
            recommend_task()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()