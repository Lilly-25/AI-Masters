# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:34:50 2024

@author: drajarajeswari
"""


import csv
import collections
# import math
import time
from io import BytesIO
import random as rn
from numpy import concatenate
from numpy import random
from numpy.random import randint
import copy
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from dateutil.relativedelta import relativedelta
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pdf2image import convert_from_path
from tkinter import filedialog, scrolledtext
from PyPDF2 import PdfReader
import webbrowser
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta


days = [("05-FEB-2024","Monday", 0), ("06-FEB-2024","Tuesday", 1), ("07-FEB-2024","Wednesday", 2), ("08-FEB-2024","Thursday", 3), ("09-FEB-2024","Friday", 4),("10-FEB-2024","Saturday", 5),("11-FEB-2024","Sunday", 6),
        ("12-FEB-2024","Monday", 7), ("13-FEB-2024","Tuesday", 8), ("14-FEB-2024","Wednesday", 9), ("15-FEB-2024","Thursday", 10), ("16-FEB-2024","Friday", 11),("17-FEB-2024","Saturday", 12),("18-FEB-2024","Sunday", 13),
        ("19-FEB-2024","Monday", 14), ("20-FEB-2024","Tuesday", 15), ("21-FEB-2024","Wednesday", 16), ("22-FEB-2024","Thursday", 17), ("23-FEB-2024","Friday", 18)]
totalDays = len(days)
examStartTiming = [("08:30",0),("11:00",1),("13:00",2),("16:00",3)]
totalExamStartTiming = len(examStartTiming)
examDuration = 1.5
courses =[]
classRooms =[]
registrations = []
totalClassRoom = -1
Individual = collections.namedtuple('Population', 'chromosome value')
population_size = 0
crossover_probability, mutation_probability = 0.0, 0.0


# Class to store course

class Course:
    
    # Initialize the class
    def __init__(self,code,name,number):
        self.courseCode = code
        self.courseName = name
        self.number = number
    
    # Print Course
    def __repr__(self):
        return '({0},{1},{2})'.format(self.courseCode, self.courseName,self.number)
    
    # Check Equality 
    def __eq__(self,other):
        return self.courseName==other.courseName and self.courseCode==other.courseCode
    
# Class to store Class Room Capacity

class Room:
    
    #Initialize the class
    def __init__(self,roomNo,capacity,index):
        self.roomNumber = roomNo
        self.capacity = capacity
        self.index = index
        
    #Print Room
    def __repr__(self):
        return '({0},{1},{2})'.format(self.roomNumber,self.capacity,self.index)
    
    #Check Equality
    def __eq__(self, other):
        return self.roomNumber == other.roomNumber and self.capacity == other.capacity
  
# Class to store student registered in course

class Registration:
    
    # Initialize the class
    def __init__(self,courseCode,matrNumbers):
        
        self.registeredCourse = courseCode
        self.studentIds = matrNumbers.copy()
    
    #Print registration
    def __repr__(self):
        return '({0},{1})'.format(self.registeredCourse,self.studentIds)
    
    #Check equality
    def __eq__(self,other):
        if self.registeredCourse == other.registeredCourse and len(self.studentIds)==len(other.studentIds):
            count = 0
            for i in range(len(self.studentIds)):
                if self.studentIds[i] == other.studentIds[i]:
                    count +=1
            if count == len(self.registeredCourse):
                return True
        return False

#Class to store an exam
class Exam:
    
    #Initialize the class
    def __init__(self,course,startTime,roomNo, day):
        self.course = course
        self.startTime = startTime
        self.roomNo = roomNo.copy()
        self.day = day
        self.binary = []
    
    #Print an Exam
    def __repr__(self):
        return '(\n {0},{1},{2},{3},\n'.format(self.course.courseName,self.startTime, self.roomNo,self.day)
    
    #Check equality
    def __eq__(self,other):
        if self.course.courseCode == other.course.courseCode and len(self.roomNo) == len(other.roomNo) and self.startTime == other.startTime:
            count = 0
            count1 = 0
            for i in range(len(self.roomNo)):
               if self.roomNo[i] == other.roomNo[i]:
                   count +=count1
                    
# Reading from files
                    
# Function to handle file upload
def upload_file(file_type):
    file_path = filedialog.askopenfilename(filetypes=[(f"{file_type} files", f"*.{file_type}")])
    return file_path
                    
def upload_courses(success_label):
    course_file_path = upload_file("csv")
    if course_file_path:
        with open(course_file_path) as file:
            reader = csv.reader(file)
            count = 0
            for row in reader:
                if len(row) != 0:
                    temp_course = Course(row[0], row[1], count)
                    if temp_course not in courses:
                        courses.append(temp_course)
                        count += 1
            print(row)
        success_label.config(text="Courses upload successful!")
    else:
        success_label.config(text="Courses upload canceled or failed.")

# Function to read input files for rooms
def upload_rooms(success_label):
    room_file_path = upload_file("csv")
    if room_file_path:
        with open(room_file_path) as file:
            reader = csv.reader(file)
            next(reader, None)
            count = 0
            for row in reader:
                if len(row) != 0:
                    temp_room = Room(row[1], row[2], count)
                    if temp_room not in classRooms:
                        classRooms.append(temp_room)
                        count += 1
        success_label.config(text="Rooms upload successful!")
        global totalClassRoom
        totalClassRoom = len(classRooms)
    else:
        success_label.config(text="Rooms upload canceled or failed.")



# Function to read input files for exam registrations
def upload_exam_registrations(success_label):
    exam_file_path = upload_file("csv")
    if exam_file_path:
        with open(exam_file_path) as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                flag = False
                checkCourseCode = [x for x in courses if x.courseCode == row[1]]
                if len(row) != 0 and row[0] != '' and row[1] != '' and len(checkCourseCode):
                    temp_registration = Registration(row[1], [row[0]])
                    if len(registrations) != 0:
                        for i in registrations:
                            if i.registeredCourse == row[1] and row[0] not in i.studentIds:
                                i.studentIds.append(row[0])
                                flag = True
                                break
                    if not flag:
                        registrations.append(temp_registration)
        success_label.config(text="Exam Registrations upload successful!")
    else:
        success_label.config(text="Exam Registrations upload canceled or failed.")


                    
# Getting Top 5 registered courses
registeredCourse_count = [(r.registeredCourse,len(r.studentIds)) for r in registrations]
sortedCourse_count = sorted(registeredCourse_count, key= lambda x:x[1],reverse=True)
max_count = [r[0] for r in sortedCourse_count][:5]   
                    
    
# Genrating random exam
def getRandomExam(course):
    courseCode = course.courseCode
    total_students = -1
    
    for x in registrations:
        if courseCode == x.registeredCourse:
            total_students = len(x.studentIds)
            break
    
    #Assinging rooms required to accomodata all students
    roomNo = []
    while total_students > 0:
      temp = classRooms[rn.randrange(0,totalClassRoom)]
      while temp in roomNo:
          temp = classRooms[rn.randrange(0,totalClassRoom)]
      roomNo.append(temp)
      total_students -= int(temp.capacity)
    # print(roomNo)
    #Setting time
    startTime = examStartTiming[rn.randrange(0,totalExamStartTiming)]
    
    #Setting Day
    if course in max_count:
        day = days[rn.choice([0,2,4,7,9])]
    else :
        day = days[rn.choice([1,3,8,10,11,14,15,16,17,18])]

    return Exam(course,startTime,roomNo,day)   


def calculate_hard_constraints(chromosome):
    
    hard_constraints_violations = 0
    h1 = True
    h2 = True
    h3 = True
    h4 = True
    
    # H1: A student cannot sit more than more one exam at the same time.
    for exam in chromosome:
        
        students_exam1 = set([el for s in registrations if s.registeredCourse == exam.course.courseCode for el in s.studentIds])
        # print('student1',students_exam1)
        exam_on_sametime = [e for e in chromosome if e.day == exam.day and e.startTime == exam.startTime and e.course.courseCode != exam.course.courseCode]
        for c in exam_on_sametime:
            students_exam2 = set([el for s in registrations if s.registeredCourse == c.course.courseCode for el in s.studentIds])
            # print('student2',students_exam2)
            if students_exam2 :
                # print('Common', students_exam1 & students_exam2)
                if len((students_exam1 & students_exam2)) !=0:
                    hard_constraints_violations += 5
                    h1 = False

                # print(hard_constraints_violations)
      
                
        # H2: The capacity of the exam should not exceed the room capacity.
        room_capacity = 0
        for room in exam.roomNo:
            room_capacity += int(room.capacity)
        if room_capacity < len(students_exam1):
            hard_constraints_violations += 5
            h2 = False
        
        # H3 : No exam scheduled in the same room at the same time
        rooms_at_sametime = set([r.roomNumber for e in exam_on_sametime for r in e.roomNo])
        # print(rooms_at_sametime)
        # print('  kkk')
        # print([r.roomNumber for r in exam.roomNo])
        # print(rooms_at_sametime & set([r.roomNumber for r in exam.roomNo]))
        if len(rooms_at_sametime & set([r.roomNumber for r in exam.roomNo])) != 0:
            hard_constraints_violations += 5
            h3 = False
        
    # H4 : No exams are scheduled in Saturdays and Sundays
    exams_on_holidays = [e for e in chromosome if e.day[1] in ("Saturday","Sunday","Holiday")]
    if len(exams_on_holidays) !=0:
        hard_constraints_violations +=5 
        h4 = False
    
    return hard_constraints_violations,h1,h2,h3,h4

def calculate_soft_constraints(chromosome):
    
    soft_constraints_violations = 0
    s1 = True
    s2 = True
    s3 = True
    s4 = True
    s5 = True
    one_minus_totRooms = len(classRooms)-1 # Used in S3 
    
    # S1 Two exams in a row: minimize student sitting consecutive exams in a consecutive days.
    s1_student_count = 0
    for i in range(len(days)-1):
        if days[i] is not None and days[i+1] is not None:
            students_exam1 = set([s for e in chromosome if e.day == days[i] for r in registrations if r.registeredCourse == e.course.courseCode for s in r.studentIds])
            students_exam2 = set([s for e in chromosome if e.day == days[i+1] for r in registrations if r.registeredCourse == e.course.courseCode for s in r.studentIds])
            common_student = students_exam1 & students_exam2
            # print(len(common_student))
            s1_student_count += len(common_student)
            if len(common_student) > 10:
                # print(len(common_student))
                soft_constraints_violations += 1
                s1 = False
    # print(soft_constraints_violations)
    
    # S2 Two exams in a day: minimize student sitting more than two exams in a day.
    for day in days:
        if day is not None:
            students_exam_a_day = [r.studentIds for e in chromosome if e.day == day for r in registrations if r.registeredCourse == e.course.courseCode]
            # print(len(students_exam_a_day))
            for i in range(len(students_exam_a_day)-1):
                j = i+1
                # print(j)
                count =0
                while(j<len(students_exam_a_day)):
                    temp = set(students_exam_a_day[i]) & set(students_exam_a_day[j])
                    # print(temp)
                    j+=1
                    if len(temp)>0:
                        count+=1
                if(count >2):
                    soft_constraints_violations += 1
                    s2 = False
            
    # S3 Period penalty: minimize the number of exams scheduled in the period with a penalty.
            exam_count_per_period = 0
            for t in examStartTiming:
                exam_count_per_period = len([e for e in chromosome if e.day == day and e.startTime == t])
                # print(exam_count_per_period, one_minus_totRooms)
                if exam_count_per_period > one_minus_totRooms :
                    soft_constraints_violations += 1 
                    s3 = False
                    
     # S4 Room penalty: minimize the number of exams scheduled in a room with a penalty.
            exam_count_per_room = 0
            for r in classRooms:
                exam_count_per_room = len([e for e in chromosome if e.day == day and (r in e.roomNo)])
                # print(exam_count_per_room, totalExamStartTiming)
                if exam_count_per_room > totalExamStartTiming:
                    soft_constraints_violations += 1 
                    s4 = False
                    
    # S5 Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.   
    # student_registrations_count = 0
    # l_exams = [e.course.courseCode for e in chromosome if e.day[1] in [10,11,12,13,14]]
    # for r in registrations:
    #     if r.registeredCourse in l_exams and student_registrations_count < len(r.studentIds):
    #         student_registrations_count = len(r.studentIds)
    # # print(student_registrations_count)
    # if student_registrations_count != 0: 
    #     soft_constraints_violations += 1
    #     s5 = False
    
    lw_exams = [e.course.courseCode for e in chromosome if e.day[2] in [14,15,16,17,18]]
    temp = set(max_count) & set(lw_exams)
    # print(temp)
    if len(temp) !=0:
        soft_constraints_violations += 1
        s5 = False
    return soft_constraints_violations,s1,s2,s3,s4,s5,s1_student_count

# Calculating fitness of given chromosome
def calculate_value(chromosome):
    value = 400
    value -= calculate_hard_constraints(chromosome)[0]
    value -= calculate_soft_constraints(chromosome)[0]

    # Binary encoding
    for i in range(len(chromosome)):
        chromosome[i].binary.clear()
        chromosome[i].binary.append(bin(chromosome[i].course.number)[2:].zfill(6))
        chromosome[i].binary.append(bin(chromosome[i].startTime[1])[2:].zfill(6))
        tempRoom = []
        for room in chromosome[i].roomNo:
            tempRoom.append(bin(room.index)[2:].zfill(6))
        chromosome[i].binary.append(bin(chromosome[i].day[2])[2:].zfill(6))
    return value

# Assigning fitness to the chromosomes in population
def calculate_fitness(population):
    for i in range(len(population)):
        v = calculate_value(population[i].chromosome)
        population[i] = Individual(
            chromosome=population[i].chromosome,
            value=v
        )
    return population

#Generate Random Initial Solution
def generate_population(size):
    new_population = []
    
    #Initialize Random population
    for i in range(size):
        timetable = []
        for j in courses:
            timetable.append(getRandomExam(j))
        new_population.append(Individual(chromosome=timetable,value=-1))
    return new_population
    
#Apply Mutation on chromosomes
def apply_mutation(chromosome):
    if random.randint(0,100) <= mutation_probability*100:
        gene = random.randint(0,len(courses)-1)
        chromosome[gene] = getRandomExam(courses[gene])
    return chromosome

#Apply Crossover on population
def apply_crossover(population):
    crossover_population = []
    
    while len(crossover_population) < len(population):
        if randint(0,100) <= crossover_probability*100:
            #Selecting parents
            parent_a = randint(0,len(population)-1)
            parent_b = randint(0,len(population)-1)
            
            # Doing crossover
            chromosome_a = copy.deepcopy(concatenate((population[parent_a].chromosome[:int(len(courses) / 2)],
                                                      population[parent_b].chromosome[int(len(courses) / 2):])))
            chromosome_a = apply_mutation(chromosome_a)

            chromosome_b = copy.deepcopy(concatenate((population[parent_b].chromosome[:int(len(courses) / 2)],
                                                      population[parent_a].chromosome[int(len(courses) / 2):])))
            chromosome_b = apply_mutation(chromosome_b)

            crossover_population.append(Individual(
                chromosome=chromosome_a,
                value=-1
            ))
            crossover_population.append(Individual(
                chromosome=chromosome_b,
                value=-1
            ))

    # Calculating fitness of crossover population
    crossover_population = calculate_fitness(crossover_population)
    # Combining will all population
    population = population + crossover_population
    return population

def elitism_selection(population, elite_count):
    # Sort the population based on fitness in descending order
    sorted_population = sorted_population = sorted(population, key=lambda individual: individual.value, reverse=True)
    
    # Select the top individuals (elites) without applying crossover or mutation
    elites = sorted_population[:elite_count]
    
    return elites

# Roulette Wheel Selection
def roulette_wheel_selection(population):
    # Calculating total fitness
    population_fitness = sum([individual.value for individual in population])
    # Calculating probabilities of all chromosomes
    chromosome_probabilities = [round(individual.value / population_fitness, 5) for individual in population]

    copy_probabilities = chromosome_probabilities.copy()
    copy_probabilities.sort()
    for i in range(len(copy_probabilities)):
        if i != 0:
            copy_probabilities[i] = round(copy_probabilities[i] + copy_probabilities[i - 1], 5)

    # Selecting population
    selected_population = []
    for i in range(population_size):
        index = -1
        random_probability = round(random.uniform(0, 1), 5)
        for j in range(len(copy_probabilities)):
            if random_probability <= copy_probabilities[j]:
                value = copy_probabilities[j]
                if j != 0:
                    value = round(value - copy_probabilities[j - 1], 5)
                index = chromosome_probabilities.index(value)
                break
        selected_population.append(population[index])
    return selected_population

# Find Top Fittest Individual from Population
def find_fittest_individual(population):
    highest_value = 0
    highest_index = 0
    for i in range(len(population)):
        if population[i].value > highest_value:
            highest_value = population[i].value
            highest_index = i
    return population[highest_index]


def display_images(images):
    for img in images:
        tk_img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img


def calculate_end_time(day, start, duration_hours):
    # Convert day and start time into a single datetime object
    start_time = datetime.strptime(f"{day} {start}", "%m-%d-%Y %H:%M")
    # Calculate end time by adding duration
    end_time = start_time + timedelta(hours=duration_hours)
    return end_time
    
def generate_colors(n):
    cm = plt.get_cmap('tab20')  # Get a colormap from Matplotlib
    colors = [cm(1.*i/n) for i in range(n)]  # Generate n distinct colors
    return colors

def display_schedule_abc(best_solution,filename):
    

    best_solution_copy = copy.deepcopy(best_solution)
    best_solution_copy = Individual(sorted(best_solution_copy.chromosome, key=lambda x:(x.day[2], x.startTime[1])), best_solution_copy.value)
    
    for e in best_solution_copy :
        print(e)
    
    exam_schedule = best_solution_copy.chromosome
     # Convert Exam objects to a list of dictionaries
    # data = [
    # {"Room": ", ".join([r.roomNumber for r in exam.roomNo]), 
    #  "Course": exam.course.courseName,
    #  "Day": exam.day[0],
    #  "Start": exam.startTime[0],
    #  "Duration": 1
    # }
    #   for exam in exam_schedule
    # ]  
    # df = pd.DataFrame(data)
    # Calculate End Time
    # df['End'] = df.apply(lambda row: calculate_end_time(row['Day'], row['Start'], row['Duration']), axis=1)
    # df['Start'] = df.apply(lambda row: datetime.strptime(f"{row['Day']} {row['Start']}", "%m-%d-%Y %H:%M"), axis=1)
    # # Prepare the data for the Gantt chart
    # df_gantt = df.copy()
    # df_gantt['Task'] = df_gantt['Course']
    # df_gantt = df_gantt.rename(columns={'Room': 'Resource', 'Start': 'Start', 'End': 'Finish'})
    # df_gantt = df_gantt.head(20)
    # num_resources = len(df_gantt['Resource'].unique())  # Number of unique resources
    # color = generate_colors(num_resources)

    # # Generate the Gantt chart
    # fig = ff.create_gantt(df_gantt, colors= color, index_col='Resource', title='Exam Schedule', show_colorbar=True, group_tasks=True, task_names='Task', bar_width=0.3)


    # # Display the plot in Spyder console
    # fig.show()
    
    # # Convert string timestamps to datetime objects for proper plotting
    # df["Start"] = pd.to_datetime(df["Start"])
    # df['End'] = df.apply(lambda row: row['Start'] + timedelta(hours=1, minutes=30), axis=1)
    # # Extract day, time slot, and room information
    # df["Day"] = df["Start"].dt.date
    # df["TimeSlot"] = df["Start"].dt.strftime("%H:%M")
    # df["EndTimeSlot"] = df["End"].dt.strftime("%H:%M")
    # # Count exams per day
    # exam_counts = df["Day"].value_counts().sort_index()

    # # Plot the bar chart
    # axis[1].bar(exam_counts.index, exam_counts.values)
    # axis[1].set_xlabel("Day")
    # axis[1].set_ylabel("Number of Exams")
    # axis[1].set_title("Distribution of Exams Per Day")
    # axis[1].set_xticklabels(exam_counts.index, rotation=45, ha="right")  # Rotate x-axis labels for better visibility
    # # Create a 3D scatter plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatter plot for each exam
    # for index, row in df.iterrows():
    #     ax.scatter(row["Day"], row["Start"], row["Room"], label=row["Course"])
    # # Customize plot labels
    # ax.set_xlabel("Day")
    # ax.set_ylabel("Time Slot")
    # ax.set_zlabel("Room")
    # ax.set_title("3D Visualization of Exam Schedule")
    
    # Show the plot
    # plt.show()


    # # Convert start time to datetime objects
    # df['Start'] = df['Start'].apply(lambda x: parser.parse(x))

    # df['End'] = df.apply(lambda row: row['Start'] + timedelta(hours=1, minutes=30), axis=1)

    #  # Convert datetime to total seconds for JSON serialization
    # df['Start'] = df['Start'].apply(lambda x: x.timestamp())
    # df['End'] = df['End'].apply(lambda x: x.timestamp())

   # Calculate end time as start time plus 1.5 hours
    # df.loc[df['Start'] == '8:30','End'] = '10:00'
    # df.loc[df['Start'] == '11:00','End'] = '12:30'
    # df.loc[df['Start'] == '13:30','End'] = '15:30'
    # df.loc[df['Start'] == '16:00','End'] = '17:30'
    
    # Create a style sheet for paragraphs
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=letter)
    table_data = [["Date",'Time', "Subject", "Subject Code", "Room No"]]
    for e in exam_schedule:

        table_data.append([Paragraph(f'<b>{e.day[0]} {e.day[1]}</b>', styles['Normal']),
        Paragraph(f'<b>{e.startTime[0]}</b>', styles['Normal']),
        Paragraph(f'<b>{e.course.courseName}</b>', styles['Normal']),
        Paragraph(f'<b>{e.course.courseCode}</b>', styles['Normal']),
        Paragraph(f'<b>{", ".join([r.roomNumber for r in e.roomNo])}</b>', styles['Normal']),
])
    # Create the table with calendar-style layout
    table = Table(table_data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    table.setStyle(style)

    # Build the PDF document
    doc.build([table])




# Run Complete Algorithm Step by step
def runGA():
    # Generating random population
    population = generate_population(population_size)
    generation = 1
    best_solution = None

    # Calculate Fitness of initial population
    population = calculate_fitness(population)
    start_time = time.time()
    total_time = 600 # Approx 5 min = 300
    elite_count = 5

    # Running generations
    while True:
        
        # Apply elitism to select the top individuals
        elites = elitism_selection(population, elite_count)
        # Applying crossover and mutation
        population = apply_crossover(population)
        # Combine elites and offspring to form the next generation
        population += elites
        # Selection using roulette wheel
        population = roulette_wheel_selection(population)
        # Finding fittest candidates
        candidate = find_fittest_individual(population)

        # Updating best solution so far
        if best_solution is None:
            best_solution = candidate
        elif candidate.value > best_solution.value:
            best_solution = candidate

        # print Every 10th generation results
        if generation % 10 == 0 or generation == 1:
            print('\nCurrent generation: {}'.format(generation))
            print('Best solution so far: {}, Goal: 400'.format(best_solution.value))

        # break when solution is found
        if best_solution.value == 400 or generation == 1000: # Remaining time = (current_time - start_time)
            print('\nSolution found:')
            print('Value: {}, Goal: 400'.format(best_solution.value))
            
            _,h1,h2,h3,h4 = calculate_hard_constraints(best_solution.chromosome)

            print("\nHard Constraints:")
            print("1: An exam will be scheduled for each course\t\t\t\t ✔")
            if h1:
                print("2: A student can not give more than one exam at a time\t\t\t ✔")
            else:
                print("2: A student can not give more than one exam at a time\t\t\t ❌")
            print("3: All exams must be held between 9 AM and 5 PM\t\t\t\t ✔")
            if h2:
                print("4:The capacity of the exam should not exceed the room capacity.\t\t ✔")
            else:
                print("4: The capacity of the exam should not exceed the room capacity.\t\t ❌")
            if h3:
                print("5: No exam scheduled in the same room at the same time\t\t\t ✔")
            else:
                print("5: No exam scheduled in the same room at the same time\t\t\t ❌")
            if h4:
                print("6: No exams are scheduled in Saturdays and Sundays\t\t\t ✔")
            else:
                print("6: No exams are scheduled in Saturdays and Sundays\t\t\t ❌")
            
            _,s1,s2,s3,s4,s5,s1_student_count = calculate_soft_constraints(best_solution.chromosome)
            
            print("\nSoft Constraints:")
            print("1: An exam will be scheduled for each course\t\t\t\t ✔")
            if s1:
                print("2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ✔")
            else:
                print("2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ❌", s1_student_count)
            print("3: One hours of break for faculty meeting\t\t\t\t ✔")
            if s2:
                print("4: Two exams in a day: minimize student sitting more than two exams in a day.\t\t ✔")
            else:
                print("4: Two exams in a day: minimize student sitting more than two exams in a day.\t\t ❌")
            if s3:
                print("5: Period penalty: minimize the number of exams scheduled in the period with a penalty.\t\t ✔")
            else:
                print("5: Period penalty: minimize the number of exams scheduled in the period with a penalty.\t\t ❌")
            if s4:
                print("6: Room penalty: minimize the number of exams scheduled in a room with a penalty.\t\t ✔")
            else:
                print("6: Room penalty: minimize the number of exams scheduled in a room with a penalty.\t\t ❌")
            if s5:
                print("7: Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.\t\t ✔")
            else:
                print("7: Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.\t\t ❌")
            
            return best_solution
        
        generation += 1


def display_schedule_and_download(best_solution, filename, h1, h2, h3, h4, s1, s2, s3, s4, s5, s1_student_count):
    # Your existing display_schedule code goes here
    display_schedule_abc(best_solution, filename)
    # Open the PDF file for download
    def download_pdf():
        webbrowser.open_new(filename)

    # Create the main window
    root = tk.Tk()

    root.config(bg="lightgreen")

    root.attributes('-fullscreen', True)

    root.title("Exam Schedule Optimization")

    title_label_bold = tk.Label(root, text="THWS University Semester Exam Schedule Winter", font=("Helvetica", 16, "bold"), bg="SystemButtonFace")
    title_label_bold.pack(pady=10)

    # Add a subtitle label with italic font
    subtitle_label_italic = tk.Label(root, text="FIW-Computer Science & Business Informatics", font=("Helvetica", 12, "bold"), bg="SystemButtonFace")
    subtitle_label_italic.pack()

    image2_path = './images/image3.jpeg'
    image = Image.open(image2_path)
    image = image.resize((200, 200), Image.BICUBIC)
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.image = photo  # This line is crucial to prevent the image from being garbage collected
    image_label.pack()

    # # Add a subtitle label with italic font
    # second_subtitle_label_italic = tk.Label(root, text="Optimized Exam Calendar", font=("Helvetica", 8, "italic"))
    # second_subtitle_label_italic.pack()

    # Add a button to download the PDF
    download_button = tk.Button(root, text="Download Exam Schedule as PDF", command=download_pdf)
    download_button.pack(pady=10)

    # # Create a scrolled text widget for displaying PDF content
    # pdf_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=30)
    # pdf_display.pack(pady=10)

    # #Display PDF content in the scrolled text widget
    # with open(filename, "rb") as pdf_file:
    #     pdf_reader = PdfReader(pdf_file)
    #     #for page_num in range(pdf_reader.numPages):
    #     for page_num in range(len(pdf_reader.pages)):    
    #         page = pdf_reader.pages[page_num]
    #         pdf_display.insert(tk.END, page.extract_text())


    h_label = tk.Label(root, text="Hard Constraints:", font=("Helvetica", 10, "bold"), bg="SystemButtonFace")
    h_label.pack()
    h1_label= tk.Label(root, text="1: An exam will be scheduled for each course\t\t\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    h1_label.pack()

    if h1:
        h2_label = tk.Label(root, text="2: A student can not give more than one exam at a time\t\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        h2_label = tk.Label(root, text="2: A student can not give more than one exam at a time\t\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    h2_label.pack()

    h3_label = tk.Label(root, text="3: All exams must be held between 9 AM and 5 PM\t\t\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    h3_label.pack()

    if h2:
        h4_label = tk.Label(root, text="4:The capacity of the exam should not exceed the room capacity.\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        h4_label = tk.Label(root, text="4: The capacity of the exam should not exceed the room capacity.\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    h4_label.pack()

    if h3:
        h5_label = tk.Label(root, text="5: No exam scheduled in the same room at the same time\t\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        h5_label = tk.Label(root, text="5: No exam scheduled in the same room at the same time\t\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    h5_label.pack()

    if h4:
        h6_label = tk.Label(root, text="6: No exams are scheduled in Saturdays and Sundays\t\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        h6_label = tk.Label(root, text="6: No exams are scheduled in Saturdays and Sundays\t\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    h6_label.pack()

    s_label = tk.Label(root, text="Soft Constraints:", font=("Helvetica", 10, "bold"), bg="SystemButtonFace")
    s_label.pack()
    s1_label= tk.Label(root, text="1: An exam will be scheduled for each course\t\t\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    s1_label.pack()

    if s1:
        s2_label = tk.Label(root, text="2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        ####print("2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ❌", s1_student_count)
        #s2_label = tk.Label(root, text="2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
        s2_label = tk.Label(root, text=f"2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ❌ {s1_student_count} violations", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    s2_label.pack()

    s3_label = tk.Label(root, text="3: One hours of break for faculty meeting\t\t\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    s3_label.pack()

    if s2:
        s4_label = tk.Label(root, text="4: Two exams in a day: minimize student sitting more than two exams in a day.\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        s4_label = tk.Label(root, text="4: Two exams in a day: minimize student sitting more than two exams in a day.\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    s4_label.pack()

    if s3:
        s5_label = tk.Label(root, text="5: Period penalty: minimize the number of exams scheduled in the period with a penalty.\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        s5_label = tk.Label(root, text="5: Period penalty: minimize the number of exams scheduled in the period with a penalty.\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    s5_label.pack()

    if s4:
        s6_label = tk.Label(root, text="6: Room penalty: minimize the number of exams scheduled in a room with a penalty.\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        s6_label = tk.Label(root, text="6: Room penalty: minimize the number of exams scheduled in a room with a penalty.\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    s6_label.pack()

    if s5:
        s7_label = tk.Label(root, text="7: Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.\t\t ✔", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    else:
        s7_label = tk.Label(root, text="7: Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.\t\t ❌", font=("Helvetica", 8, "italic"), bg="SystemButtonFace")
    s7_label.pack()

    close_button = tk.Button(root, text="Exit", command=root.destroy)
    close_button.pack(pady=10)

    # image2_path = './images/image3.jpeg'
    # image = Image.open(image2_path)
    # image = image.resize((200, 200), Image.BICUBIC)
    # photo = ImageTk.PhotoImage(image)
    # image_label = tk.Label(root, image=photo)
    # image_label.image = photo  # This line is crucial to prevent the image from being garbage collected
    # image_label.pack()

    root.mainloop()


if __name__ == "__main__":
    # Taking input from files
    # Create the main window
    root = tk.Tk()

    root.title("Exam Schedule Optimization")

    root.config(bg="orange")

    root.attributes('-fullscreen', True)

    title_label_bold = tk.Label(root, text="THWS University Semester Exam Schedule", font=("Helvetica", 16, "bold"), bg="orange")
    title_label_bold.pack(pady=10)

    # Add a subtitle label with italic font
    subtitle_label_italic = tk.Label(root, text="FIW-Computer Science & Business Informatics", font=("Helvetica", 12, "bold"), bg="orange")
    subtitle_label_italic.pack()

    image1_path = './images/image1.png'
    image = Image.open(image1_path)
    image = image.resize((200, 200), Image.BICUBIC)
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.image = photo  # This line is crucial to prevent the image from being garbage collected
    image_label.pack()

    # Add a subtitle label with italic font
    second_subtitle_label_italic = tk.Label(root, text="Upload files", font=("Helvetica", 8, "italic"), bg="orange")
    second_subtitle_label_italic.pack()

    root.title("File Upload")

    # Buttons for each type of file
    courses_button = tk.Button(root, text="Upload Courses", command=lambda: upload_courses(success_label_courses))
    courses_button.pack(pady=10)

    success_label_courses = tk.Label(root, text="")
    success_label_courses.pack()

    rooms_button = tk.Button(root, text="Upload Rooms", command=lambda: upload_rooms(success_label_rooms))
    rooms_button.pack(pady=10)

    success_label_rooms = tk.Label(root, text="")
    success_label_rooms.pack()

    examregs_button = tk.Button(root, text="Upload Exam Registrations", command=lambda: upload_exam_registrations(success_label_examregs))
    examregs_button.pack(pady=10)

    success_label_examregs = tk.Label(root, text="")
    success_label_examregs.pack()

    # Add a button to close the display window
    close_button = tk.Button(root, text="Generate the Exam Schedule", command=root.destroy)
    close_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", command=root.destroy)
    exit_button.pack(pady=10)

    # Start the Tkinter event loop
    root.mainloop()

    
    # Initializing population size and Crossover and Mutation Probabilities
    population_size = 100 #random.randint(50, 100)
    crossover_probability = 0.6#round(random.uniform(low=0.3, high=1.0), 1)
    mutation_probability = 0.2#round(random.uniform(low=0.0, high=0.5), 1)

    # Printing Initialized variables
    print('----- Generated Parameters -----')
    print('Population size......: {}'.format(population_size))
    print('Crossover probability: {}'.format(crossover_probability))
    print('Mutation probability.: {}'.format(mutation_probability))

    # Running Genetic Algorithm
    best_solution = runGA()

    _,h1,h2,h3,h4 = calculate_hard_constraints(best_solution.chromosome)
    _,s1,s2,s3,s4,s5,s1_student_count = calculate_soft_constraints(best_solution.chromosome)

    display_schedule_and_download(best_solution, "exam_schedule_calendar9.pdf", h1, h2, h3, h4, s1, s2, s3, s4, s5, s1_student_count)

   
