import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from itertools import combinations
from fpdf import FPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

def initialize_schedule(df_examreg, df_room, num_days, num_time_slots):
    global course_student_counts, room_capacity, num_rooms

    course_student_counts = df_examreg.groupby('Course Code')['Student ID'].nunique().to_dict()

    room_capacity = {0: 63, 1: 32, 2: 31,  3: 27, 4: 24, 5: 24, 6: 18, 7: 16, 8: 16, 9: 16, 10: 15, 11: 10, 12: 12}
    
    num_rooms = len(room_capacity.keys()) 

    # Create a 3D matrix filled with empty lists
    schedule = [[[0 for _ in range(num_rooms)] for _ in range(num_time_slots)] for _ in range(num_days)]

    day = 0

    courseids = list(sorted(course_student_counts.keys(), key=lambda x: course_student_counts[x], reverse=True)) #Sort courses so courses with greater count of registered student ids are scheduled first

    # print(courseids)
    
    # print(len(courseids))
    
    common_day_student_ids = set() 
    exams_consecutive_days=0

    for day in range(num_days):

        skipped_courses = []  # To keep track of courses which have common students in an already scheduled exam

        # if exams_consecutive_days%2 == 0:
        #     common_day_student_ids = set() 
        common_day_student_ids = set() 
        for time_slot in range(num_time_slots):
            
            room_capacity_copy = room_capacity.copy()   # to refresh rooms in list after a timeslot is scheduled
            courses_per_timeslot = 1

            while courseids and room_capacity_copy:

                course_code = courseids[0]
                student_count = course_student_counts[course_code]

                remaining_students = student_count

                course_student_ids = set(df_examreg[df_examreg['Course Code'] == course_code]['Student ID'])

                if not common_day_student_ids.intersection(course_student_ids):

                    while remaining_students > 0:    # when student count exceeds the room capacity count

                        available_rooms = list(sorted(room_capacity_copy.keys(), key=lambda x: room_capacity_copy[x], reverse=True))    #Sort rooms so larger capcity rooms are scheduled first

                        if available_rooms:

                            room_id = available_rooms[0]
                            schedule[day][time_slot][room_id] = course_code

                            remaining_students -= room_capacity_copy[room_id]
                            del room_capacity_copy[room_id]

                        else:
                            #Makes sure courses if split are scheduled in different rooms as per capacity but within same timeslot
                            skipped_courses.append(courseids.pop(0))
                            for room_id_index in range(num_rooms):
                                if schedule[day][time_slot][room_id_index] == course_code:
                                    schedule[day][time_slot][room_id_index] = 0
                            remaining_students = student_count
                            break

                    course_student_counts[course_code] = remaining_students

                    if remaining_students <= 0:

                        course_student_counts.pop(course_code)
                        courseids.remove(course_code)

                    common_day_student_ids.update(course_student_ids)
                    
                    courses_per_timeslot+=1
                    
                else:
                    skipped_courses.append(courseids.pop(0))
                    if schedule[day][time_slot][room_id] == course_code:
                        del schedule[day][time_slot][room_id]
                        
                if courses_per_timeslot >= 6: # Limiting the courses scheduled to be max 6 within a timeslot
                    break
        courseids.extend(skipped_courses)
        exams_consecutive_days+=1
    print(courseids)
    return schedule


def upload_examreg_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        df_examreg = pd.read_csv(file_path)  
        return df_examreg
    else:
        return None


def upload_room_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        df_room = pd.read_csv(file_path)
        return df_room  
    else:
        return None

def upload_course_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        df_courses = pd.read_csv(file_path)
        return df_courses
    else:
        return None


def generate_timetable():
    global initial_schedule, df_courses, df_examreg, df_room, num_days, num_time_slots

    # Get exam registration data
    df_examreg = upload_examreg_file()
    if df_examreg is None:
        return  

    # Get room data
    df_room = upload_room_file()
    if df_room is None:
        return  

    # Get course data
    df_courses = upload_course_file()
    if df_courses is None:
        return  
    
    # Get number of days and time slots
    num_days = int(num_days_entry.get())
    num_time_slots = int(num_time_slots_entry.get())
 
    # Initialize the schedule
    initial_schedule = initialize_schedule(df_examreg, df_room, num_days, num_time_slots)

    # Generate PDF
    generate_pdf(initial_schedule, df_examreg, df_room, df_courses)




def generate_pdf(initial_schedule, df_examreg, df_room, df_courses):
    timeslots = {0: '8:00 - 10:00', 1: '11:00 - 13:00', 2: '14:00 - 16:00', 3: '15:00 - 17:00'}
    days = {0: 'Monday(1/29)', 1: 'Tuesday(1/30)', 2: 'Wednesday(1/31)', 3: 'Thursday(2/1)', 4: 'Friday(2/1)', 5: 'Monday(2/5)', 6: 'Tuesday(2/6)', 7: 'Wednesday(2/7)', 8: 'Thursday(2/8)', 9: 'Friday(2/9)',
            10: 'Monday(2/12)', 11: 'Tuesday(2/13)', 12: 'Wednesday(2/14)', 13: 'Thursday(2/15)', 14: 'Friday(2/16)', 15: 'Monday(2/19)', 16: 'Tuesday(2/20)', 17: 'Wednesday(2/21)', 18: 'Thursday(2/22)', 19: 'Friday(2/23)'}
    
    # Initialize FPDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    data = []
    for day, day_schedule in enumerate(initial_schedule):
        for time_slot, time_slot_schedule in enumerate(day_schedule):
            for room_id, course_id in enumerate(time_slot_schedule):
                # Look up course name from df_courses
                course_info = df_courses[df_courses['Course id'] == course_id]
                if not course_info.empty:
                    course_name = course_info['Course name'].values[0]
                else:
                    if course_id == 0:
                        course_name = 'No exam'
                    else:
                        course_name = str(course_id)

                # Look up room name from df_rooms
                room_name = df_room[df_room['Room id'] == room_id]['Room Name'].values[0]
                timeslot_name = timeslots[time_slot]
                day_name = days[day]
                if course_id != 0:
                    data.append([day_name, timeslot_name, room_name, course_name, course_id])

    df_report = pd.DataFrame(data, columns=['Day', 'Time Slot', 'Room Name', 'Course Name', 'Course ID'])

    # Generate PDF report with tables
    create_pdf_report(df_report)

def create_pdf_report(df):
    # Define PDF file path
    pdf_file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if not pdf_file_path:
        return  # If the user cancels the save dialog, return without creating the PDF
    
    # Create SimpleDocTemplate object
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    
    # Create Table object
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data)

    # Add style to table
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    table.setStyle(style)

    # Add table to the PDF
    elements = [table]
    doc.build(elements)

# Example usage:
# generate_pdf(initial_schedule, df_examreg, df_room, df_courses)
       
    




    
def exit_program():
    root.destroy()

def check_hard_constraints(schedule):
    hard_constraints_violations = 0
    h1_hard_constraints = 0
    h2_hard_constraints = 0
    h3_hard_constraints = 0
    h4_hard_constraints = 0
    courses_scheduled=[]
    
    #make it global
    #room_capacity = {0: 63, 1: 32, 2: 31,  3: 27, 4: 24, 5: 24, 6: 18, 7: 16, 8: 16, 9: 16, 10: 15, 11: 10, 12: 12}
    for day, timeslots in enumerate(schedule):
        
        h2_room_capacity_violation = 0
        
        for time_slot, rooms in enumerate(timeslots):
            
            students_by_course = {}  
            course_capacity_by_room = {}
            h1_common_students_violations = 0
            exam_rooms_set = set()
            room_id = 0
            courses_scheduled_timeslot =[]
            
            for course_code in rooms:
                h3_room_timeslot=0
                # H1: A student cannot sit more than one exam at the same time.
                
                if course_code not in students_by_course and course_code!=0:
                    
                    existing_students = students_by_course.get(course_code, set())
                    new_students = set(df_examreg[df_examreg['Course Code'] == course_code]['Student ID'])
                    students_by_course[course_code] = existing_students.union(new_students)
                    
                # H2: The capacity of the exam should not exceed the room capacity.
                
                room_capacity_value = room_capacity.get(room_id, 0)

                if course_code in course_capacity_by_room:
                    course_capacity_by_room[course_code].append(room_capacity_value)
                else:
                    course_capacity_by_room[course_code] = [room_capacity_value]
                
                # CHECK H3 : No exam scheduled in the same room at the same time   
                # Check if the room_id has already been seen
                if room_id in exam_rooms_set:
                    #print(f"Constraint violation for unique Rooms in a timeslot: Room {room_id} is assigned to multiple courses.")
                    h3_room_timeslot+=  1
                else:
                    exam_rooms_set.add(room_id)
                    
                room_id+=1
                
                if course_code == 0 or course_code in courses_scheduled_timeslot:
                    pass
                else:
                    courses_scheduled_timeslot.append(course_code)
            
            courses_scheduled.extend(courses_scheduled_timeslot)    
                
            # CHECK H1: A student cannot sit more than one exam at the same time.        
            for keys_combination in combinations(students_by_course.keys(), 2):
                sets_to_compare = [students_by_course[key] for key in keys_combination]

                common_students = set.intersection(*sets_to_compare)

                if common_students:
                    h1_common_students_violations += len(common_students)

            # if h1_common_students_violations:
            #     print("Constraint violation for exam scheduled for common students simultaneously", h1_common_students_violations)
                #hard_constraints_violations+= h1_common_students_violations * 10
            
            #CHECK H2: The capacity of the exam should not exceed the room capacity.
            for course_code, capacities in course_capacity_by_room.items():
            # Get the count of students registered for the course
                student_count = course_student_counts.get(course_code, 0)

                # Check if the sum of capacities is greater than the student count
                if student_count > sum(capacities):
                    #print(f"Constraint violation for course room capacity {course_code}: Sum of student count exceeds capacity.")
                    h2_room_capacity_violation+=  1
        # print(f'Day {day + 1} - H1 Violations',h1_common_students_violations)          
        # print(f'Day {day + 1} - H2 Violations',h2_room_capacity_violation)
        # print(f'Day {day + 1} - H3 Violations',h3_room_timeslot)  
        hard_constraints_violations+= h1_common_students_violations + h2_room_capacity_violation + h3_room_timeslot
        h1_hard_constraints+= h1_common_students_violations
        h2_hard_constraints+= h2_room_capacity_violation
        h3_hard_constraints+= h3_room_timeslot
    print('Total H1 violations: (A student cannot sit more than one exam at the same time)', h1_hard_constraints)
    print('Total H2 violations: (The capacity of the exam should not exceed the room capacity.)', h2_hard_constraints)
    print('Total H3 violations: (No exam scheduled in the same room at the same time)', h3_hard_constraints)
    duplicate_courses_scheduled = {x for x in courses_scheduled if courses_scheduled.count(x) > 1}
    h4_hard_constraints=len(duplicate_courses_scheduled)
    print('Total H4 violations: (No duplicate courses are scheduled)', h4_hard_constraints)
    return h1_hard_constraints, h2_hard_constraints, h3_hard_constraints, h4_hard_constraints
from collections import Counter

def check_soft_constraints(schedule):
    soft_constraints_violations = 0
    s1_soft_constraints = 0
    s2_soft_constraints = 0
    s6_soft_constraints=0
    
    all_courses_scheduled=[]

    students_last_day = {}  # Dictionary to store the last scheduled day for each student

    for day, timeslots in enumerate(schedule):
        
        s1_daily_students_violations = 0
        s2_daily_consecutive_violations = 0
        s6_max_courses_timselot_violations = 0

        for _, rooms in enumerate(timeslots):
            
            unique_course_codes = set(course_code for course_code in rooms if course_code != 0) # Coz, courses can be split accross multiple rooms to satisfy room capacities constraints
            all_courses_scheduled.extend(unique_course_codes)
            for course_code in unique_course_codes:
                # Track the last scheduled day for each student
                students = set(df_examreg[df_examreg['Course Code'] == course_code]['Student ID'])
                for student_id in students:
                    last_day = students_last_day.get(student_id, None)
                    #print(last_day, student_id)
                    if last_day == day :
                        # CHECK S1 More than 1 exam in a day: minimize student sitting consecutive exams on the same day.
                        s1_daily_students_violations += 1
                    elif last_day == day - 1:
                        #print(last_day, student_id)
                        # Student has an exam on consecutive days
                        s2_daily_consecutive_violations += 1
                    students_last_day[student_id] = day

            if len(unique_course_codes) > 6:
                s6_max_courses_timselot_violations+=1   
                
        # print(f'Day {day + 1} - S1 Violations:', s1_daily_students_violations)
        # print(f'Day {day + 1} - S2 Violations:', s2_daily_consecutive_violations)

        soft_constraints_violations += s1_daily_students_violations + s2_daily_consecutive_violations
        s1_soft_constraints+= s1_daily_students_violations
        s2_soft_constraints+= s2_daily_consecutive_violations
        s6_soft_constraints+=s6_max_courses_timselot_violations
        #print('Soft constraint violation', soft_constraints_violations)
    # print('Total S1 violations', s1_soft_constraints)
    # print('Total S2 violations', s2_soft_constraints)
    # print('Total S6 violations', s6_soft_constraints)

    #if len(ideal_order_schedule) == len(all_courses_scheduled):
        #print("All courses are scheduled")
        
    #t(all_courses_scheduled)
    return s1_soft_constraints, s2_soft_constraints, s6_soft_constraints


def update_constraint_labels():
    # Provide the schedule data here
    schedule =  initial_schedule
    h1, h2, h3, h4 = check_hard_constraints(schedule)
    s1, s2, s6 = check_soft_constraints(schedule)
    #constraints_label.config(text=f"H1 (A student cannot sit more than one exam at the same time): {h1} violations\nH2 (The capacity of the exam should not exceed the room capacity): {h2} violations\nH3 (No exam scheduled in the same room at the same time) : {h3} violations\nH4 (No duplicate courses are scheduled): {h4} violations\nS1 (minimize student sitting consecutive exams on the same day): {s1} violations\nS2 (minimize student sitting consecutive exams in  consecutive days): {s2} violations\nS3 (When an exam is split between multiple rooms, all those exams are scheduled at the same timeslot): 0 violations\nS4 (Exams with higher registrations are scheduled first): 0 violations\nS5 (Minimize the no of exams scheduled at the same timeslot): {s6} violations")
    #constraints_label.config(text=f"S3 (When an exam is split between multiple rooms, all those exams are scheduled at the same timeslot): 0 violations\nS4 (Exams with higher registrations are scheduled first): 0 violations\nS5 (Minimize the no of exams scheduled at the same timeslot): {s6} violations")
    #constraints_label.config(text=f"<html><b>H1 A student cannot sit more than one exam at the same time:</b></html> {h1} violations\n<html><b>H2 The capacity of the exam should not exceed the room capacity:</b></html> {h2} violations\n<html><b>H3 No exam scheduled in the same room at the same time:</b></html> {h3} violations\n<html><b>H4 No duplicate courses are scheduled:</b></html> {h4} violations\n<html><b>S1 minimize student sitting consecutive exams on the same day:</b></html> {s1} violations\n<html><b>S2 minimize student sitting consecutive exams in consecutive days:</b></html> {s2} violations\n<html><b>S3 When an exam is split between multiple rooms, all those exams are scheduled at the same timeslot:</b></html> 0 violations\n<html><b>S4 Exams with higher registrations are scheduled first:</b></html> 0 violations\n<html><b>S5 Minimize the no of exams scheduled at the same timeslot:</b></html> {s6} violations")
    constraints_label.config(text=f"H1 A student cannot sit more than one exam at the same time: {h1} violations\nH2 The capacity of the exam should not exceed the room capacity: {h2} violations\nH3 No exam scheduled in the same room at the same time : {h3} violations\nH4 No duplicate courses are scheduled: {h4} violations\nS1 minimize student sitting consecutive exams on the same day: {s1} violations\nS2 minimize student sitting consecutive exams in consecutive days: {s2} violations\nS3 When an exam is split between multiple rooms, all those exams are scheduled at the same timeslot: 0 violations\nS4 Exams with higher registrations are scheduled first: 0 violations\nS5 Minimize the no of exams scheduled at the same timeslot: {s6} violations", font=("Arial", 12, "bold"))

# Create the main window
root = tk.Tk()
root.title("Exam Timetable Generator")

# Set background color to orange
root.configure(bg="orange")

# Label and entry for number of days
num_days_label = ttk.Label(root, text="Number of Days:", background="orange", font=("Arial", 16))
num_days_label.pack(pady=10)
num_days_entry = ttk.Entry(root, font=("Arial", 16))
num_days_entry.pack(pady=5)

# Label and entry for number of time slots
num_time_slots_label = ttk.Label(root, text="Number of Time Slots:", background="orange", font=("Arial", 16))
num_time_slots_label.pack(pady=10)
num_time_slots_entry = ttk.Entry(root, font=("Arial", 16))
num_time_slots_entry.pack(pady=5)

# Create buttons to upload the exam registration file and the room data file
examreg_button = ttk.Button(root, text="Upload Exam Registration File", command=generate_timetable, style="C.TButton")
examreg_button.pack(pady=10)
room_button = ttk.Button(root, text="Upload Room Data File", command=generate_timetable, style="C.TButton")
room_button.pack(pady=5)
course_button = ttk.Button(root, text="Upload Course Data File", command=generate_timetable, style="C.TButton")
course_button.pack(pady=5)

# Button to generate initial schedule as a PDF
pdf_button = ttk.Button(root, text="Generate Initial Schedule as PDF", command=generate_pdf, style="C.TButton")
pdf_button.pack(pady=5)

# Button to check constraints
check_constraints_button = ttk.Button(root, text="Check Constraints", command=update_constraint_labels, style="C.TButton")
check_constraints_button.pack(pady=5)

# Label to display constraints
constraints_label = ttk.Label(root, text="", font=("Arial", 12), background="orange")
constraints_label.pack(pady=10)

# Button to exit the program
exit_button = ttk.Button(root, text="Exit", command=exit_program, style="C.TButton")
exit_button.pack(pady=5)

# Styling for buttons
style = ttk.Style()
style.configure("C.TButton", font=("Arial", 14))

# Run the GUI main loop
root.mainloop()