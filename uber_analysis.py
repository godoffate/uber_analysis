"""Project: Data Analysis
1) Importing the libraries
2) Loading the dataset 
3) Data cleaning and Data preprocessing
4) Exploratory data analysis"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = pd.read_csv("D:\data_science\project_1\my_uber_drives_2016.csv")

# Overview of the data
print(x)
print()
print(x.head)
print()
print(x.tail)
print()
print(x.shape)
print()
print(x.columns)

"""Lets start claning the column names without having * at the end."""

# here x.columns means "columns" are the index/series of x and is converted to string using "str" and replaced by "replace"
x.columns = x.columns.str.replace("*","")   # string.replaces("x","y") --> replaces in string itself & it does not take "inplace" argument
print(x.columns)
print(x)

"""Now, lets see how many NaN values are present in the data and where are they"""

print(x.isnull().sum())                                      # total no. NaN in each column
print(x.isnull().sum().sum())                                # total no. od NaN in the data
print((x["PURPOSE"].isnull().sum()/x.shape[0])*100)        # gives how much percentage of "PURPOSE" column is NaN values --> (503/1156)*100

"""Now, lets first solve NaN value columns which has least NaN values."""

print(x[x["END_DATE"].isnull()])     # same row
print(x[x["CATEGORY"].isnull()])     # same row 
print(x[x["START"].isnull()])        # same row 
print(x[x["STOP"].isnull()])         # same row

"""If we see above 1155 row, every column  is has NaN. So, it will be no problem to remove that 1 row."""

x.dropna(axis=0,how="all",subset=["END_DATE","CATEGORY","START","STOP"],inplace=True)     # how="all" --> row where all the given subsets have NaN will be deleted  [or]  dropna(axis=0,how="any",subset=["END_DATE"])
print(x.isnull().sum())

"""Now, lets solve the "PURPOSE" NaN values"""

x["PURPOSE"].fillna(method="ffill",inplace=True)
print(x.isnull().sum())

"""Now, lets check the types of all the values in the column and do math to the non-object columns"""

print(x.info())         # info() --> shows the type values in each column 
print(x.describe())     # describes the non-object values

# in above "MILES" column there is obviously are outliers

"""Now, lets start with "START_DATE & END_DATE ,i.e, convert the object to datetime format"""

# pd.to_datetime(column_name,format="") --> converts objects/strings into datetime.

# 1) if you want the string(object) format (YYYY/MM/DD), then format="%Y/%m/%d" can be used.
# 2) if you want the string(object) format "yyyymmdd", then format='%y%m%d' can be used.
# errors= --> "coerce" --> when a column throws error (when invalid input) then give "NaT" and continue the running of program.

# If you don't want format the datetime, just leave it and just convert "START_DATE" into "datetime" which can be formattable..

x["START_DATE"] = pd.to_datetime(x["START_DATE"],format="%m/%d/%Y %H:%M",errors="coerce")
x["END_DATE"] = pd.to_datetime(x["END_DATE"],format="%m/%d/%Y %H:%M",errors="coerce")
print(x[["START_DATE","END_DATE"]])


"""There is some data cleaning but let start with the data analysis.
Q) what is the most frequent start location?"""

print(x.groupby(["START"])["MILES"].count())          # gives the count() of each place (not repeated, as it is grouped by groupby()) in just "MILES" column
print(x.groupby(["START"]).count())                   # gives the count() of each place (not repeated, as it is grouped by groupby()) in all the columns except "START"
print(x.groupby(["START"])["START"].count())          # x.count() --> counts the no. of times it repeates
print(x.groupby(["START"])["START"].count().max())  

# [OR]     ** MOST USED METHOD**
print(x["START"].value_counts())           # x.value.counts() --> gives the count of each column in decesnding order.

"""Q) find the top 10 most frequent start location"""

print(x["START"].value_counts().head(10))

x["START"].value_counts().plot(kind="pie")    # if you want to plot in a pie chart
x["START"].value_counts().head(10).plot(kind="bar")     # if you want to plot in a bar chart

"""Same as that find the most 10 frequent stop point location as well"""

x["STOP"].value_counts().head(10)
x["STOP"].value_counts().head(10).plot(kind="bar")

"""Q) Usually people are booking for how many miles"""

x["MILES"].value_counts().head(10)
x["MILES"].value_counts().head(10).plot(kind="bar")

# [ALSO]

x["PURPOSE"].value_counts()
x["PURPOSE"].value_counts().head(10).plot(kind="bar")

"""How we can customize the bar picture. It can de done with seaborn"""

plt.figure(figsize=(15,6))            # figure(figsize=(row_width,column_width))  --> used when names of 1 column catagories merges themselves due to no space.
sns.countplot(x=x["PURPOSE"],data=x)

"""Now, lets see the countplot for catagory column"""

sns.countplot(x=x["CATEGORY"],data=x)

"""Q) make new columns of 
1) find duration of trip
2) find no. of round trips (startloc=stoploc)
3) months majority booking is done"""

# Adding of new Columns -->
# 1) x[column_name] = [values] --> adds at last
# 2) x.insert(column_index,column_name,[values]) --> adds at given position

x["MINUTES"] = x["END_DATE"] - x["START_DATE"]         # As the datetime is in the format, we can do arithmatic operations on them
# lets make it that only minutes are shown.
x["MINUTES"] = x.MINUTES.dt.total_seconds()/60      # converts "x.MINUTES", a series which has attribute "dt" into seconds using total_seconds() method and divide by 60 to get minutes.
 
 # Subquestions: create dataframe with 4 columns pur,mil,min mil, max, mil
 
# agg() method allows you to apply a function or a list of function names to be executed along one of the axis of the DataFrame, default 0, which is the index (row) axis.
# reset_index()  --> makes it to the default index format instead of some column acting as index which can be done by "set_index()" 
x.groupby(["PURPOSE"]).agg({"MILES":["mean","max","min"]}).reset_index()

# [OR]
pd.DataFrame({"MEAN": x.groupby(["PURPOSE"])["MILES"].mean(),
              "MAX": x.groupby(["PURPOSE"])["MILES"].max(),
              "MIN": x.groupby(["PURPOSE"])["MILES"].min()}).reset_index()

# Now, lets do the plot of above.
plt.figure(figsize=(17,9))       # [ALWAYS TOP OF FIGURE] here even if we give space by increasing size, we are getting merges.
plt.xticks(rotation=45)          
plt.subplot(1,2,1)
sns.boxplot(x=x["PURPOSE"],y=x["MILES"],data=x)
plt.xticks(rotation=45)           # xticks() --> can rotate ticks/labels of x axis in any angles.
plt.subplot(1,2,2)
sns.boxplot(x=x["PURPOSE"],y=x["MINUTES"],data=x)    
plt.xticks(rotation=45) 
    
    
"""[AND]"""

x[x["START"] == x["STOP"]].value_counts().sum()      # gives total no. of rounders

# Now, to make column lets define a function

def round(x):
    if x["START"] == x["STOP"]:
        return "YES"
    else:
        return "NO"
    
x["ROUND_TRIPS"] = x.apply(round,axis=1)     # apply() --> helps to apply functions in each columns

# lets make a plot for it
sns.countplot(x=x["ROUND_TRIPS"],order=x["ROUND_TRIPS"].value_counts().index,data=x)    # order= --> sorts the values in decending order. 'value_counts" gives already the decending order and ".index" means take the index of it i.e, lables but not values


"""[AND]"""

x["MONTH"] = pd.DatetimeIndex(x["START_DATE"]).month    # .month --> converts to month 

# Now, lets conver month no. to month names

dic = {1:"JANUARY",2:"FEBUARY",3:"MARCH",4:"APRIL",5:"MAY",6:"JUNE",7:"JULY",8:"AUGUST",9:"SEPTEMBER",10:"OCTOBER",11:"NOVEMBER",12:"DECEMBER"}

x["MONTH"] = x["MONTH"].map(dic)    # map() --> Map values of Series according to an input mapping or function.

"""Q) Create barplot cab booking frequency over months(in a way month higher cab comes first) """

# Pallete= --> gives the color maps to the bars. eg: default, crest, coolwarm..

sns.countplot(x=x["MONTH"],order=x["MONTH"].value_counts().index,palette="crest",data=x)      # order= --> sorts the values in decending order. 'value_counts" gives already the decending order and ".index" means take the index of it i.e, lables but not values
plt.xticks(rotation=45)

"""Q) analysis of which month the round trips are bookes"""

plt.figure(figsize=(15,6))          # can increase/ decrease length(row)/width(column) of the figure
sns.countplot(x=x["MONTH"],hue=x["ROUND_TRIPS"],data=x,)
plt.xticks(rotation=45)

"""Q) subplot of line and scatter plots of miles vs minutes"""

# can do with matplotlib too but scatterplot --> scatter & lineplot --> line. 
# Always use seaborn for creating plots as it is more visualizing and user-friendly.

plt.figure(figsize=(20,6))          # zooms out
plt.subplot(1,2,1)
sns.scatterplot(x=x["MILES"],y=x["MINUTES"],data=x)     # sns.scatterplot(x,y,size="size_of_marker",marker="marker_type",data=) 
plt.title("SCATTER PLOT")
plt.xlabel("MILES")
plt.ylabel("MINUTES")

plt.subplot(1,2,2)
sns.lineplot(x=x["MILES"],y=x["MINUTES"],data=x)
plt.title("LINE PLOT")
plt.xlabel("MILES")
plt.ylabel("MINUTES")

"""countplot --> dodge= --> compares 2 variables by "hue" in same bin/bar """

plt.figure(figsize=(15,10))
sns.countplot(x=x["PURPOSE"],hue=x["CATEGORY"],dodge=False,data=x)
plt.xticks(rotation=45)

"""Q) which catagory is travelling for more distance"""

# x.groupby(["CATEGORY"])["MILES"].sum()   --> gives tabular form of below barplot with numerics

sns.barplot(x=x["CATEGORY"],y=x["MILES"],data=x)

# [OR]
x.groupby(["CATEGORY"])["MILES"].mean().plot(kind="bar")

"""Q) purpose of trips in particular months"""

plt.figure(figsize=(15,7))
sns.countplot(x=x["PURPOSE"],hue=x["MONTH"],palette="crest",data=x)
plt.xticks(rotation=45)

# .rename({}"MILES":"distance"},axis=1) --> renames the column easily of a given DataFrame

"""Q) time where majority of cabs are booked"""

x["DAY_HOUR"] = pd.DatetimeIndex(x["START_DATE"]).hour           # .hour --> converts datetimeindex to hour

def time(x):
    if 6 >= x["DAY_HOUR"] < 12:
        return "MORNING"
    elif 12 >= x["DAY_HOUR"] < 18:
        return "AFTERNOON"
    elif 18 >= x["DAY_HOUR"] < 24:
        return "EVENING"
    else:
        return "NIGHT"

x["DAY_SLOTS"] = x.apply(time,axis=1)         # apply() --> used to apply functions to the DataFrame columns (not Series/list but 1 column dictionary)

sns.countplot(x=x["DAY_SLOTS"],data=x)






"""REPORT:
    1) cabs are usually booked for business category
    2) more round trips are done in month of dec
    3) cary is most frequent start point
    4) cary is most frequent stop point
    5) location with least booking can be offered a discount coupens
    6) more bookings are done in the january month
    7) majority of trips are not round
    8) airport bookings are only for business class
    9) charity, commute and moving is only for personal purpose
    10) majority of cabs are booked for meeting purpose
    11) business trips are for longer distance
    12) for cab bookings seasonal pattern is observed
    13) evening has max bookings and morning has min bookings
    """