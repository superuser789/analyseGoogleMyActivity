
import json
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import matplotlib
import calendar
import os
import math
import base64
import zipfile

import warnings
warnings.filterwarnings("ignore")

# print(' '.join(pytz.country_timezones['in'])) # View Timezone of a Country

Steps='''
    1. Go to : https://takeout.google.com/?pli=1
    2. Under  "Select data to include", Click on "Deselect all"
    3. Scroll Down and Select "My Activity" . Click on "Multiple formats".   In  "Activity records, Choose 'JSON'  & then 'ok'.
    4. Scroll Down. Click on "Next Step". and then on "Create Export".
    5. Wait for the Google Data Download mail to arrive in your Gmail. Download the Zip file.
'''



ReportName='Android-Activity'+'-'+datetime.now().strftime("-%m-%d-%H%M")+'/'

#os.makedirs(directory) if not os.path.exists(ReportName) else pass
if not os.path.isdir(ReportName):
    os.mkdir(ReportName)
    print("\n\n -- Saving Reports in '"+ReportName[:-1]+"' ..\n",end='')


def creategDataDF(File=''):
    '''Reads Android Activity JSON File and Returns it as Dictionary'''
    
    def unzip(file):
        extractDir=file.split('.zip')[0]
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(extractDir)
        return extractDir
    
    DefaultFileLoc='/Takeout/My Activity/Android/MyActivity.json'
    if File:
        if '.json' in File.lower(): # If Default : json file
            ActivityFile=File
        elif '.zip' in File  : # If zip file
            ActivityFile=unzip(File)+DefaultFileLoc
        else: # File path is passed
            File = File if '/' is File[-1] else File+'/'
            files=[ File+file for file in os.listdir(File) ] # Get All Files present in the directory
            latestfiles=sorted(files, key = os.path.getctime, reverse=True)[0] # Get the Latest File
            #print('\n   --- Extracted  '+latestfiles+'\n')
            ActivityFile=unzip(latestfiles)+DefaultFileLoc  
    else:
        RecentTakeoutFile = sorted([file for file in os.listdir() if 'takeout' in file.lower() and '.zip' in file], key = os.path.getctime, reverse=True)[0] # Gives the Latest Zip present in the directory
        if RecentTakeoutFile:
            #print('\n   --- Extracted  '+RecentTakeoutFile+'\n')
            ActivityFile=unzip(RecentTakeoutFile)+DefaultFileLoc        
        else:
            ActivityFile='MyActivity.json'
    if os.path.isfile(ActivityFile):
        print('\n   -- Using Activity Data File : '+ActivityFile)
        with open(ActivityFile, encoding='utf8') as file:
           gData = json.load(file)
        return gData
    else:
        print('\n\n    --- Check File Path and Try Again.\n\n')
        exit()



def makeDir(dirpath): # Creates The Directory Tree if not existed
   from pathlib import Path
   p = Path(dirpath)
   p.mkdir(exist_ok=True, parents=True)


def appDataFrame(gData):
    '''
    Using Google Data Dictionary, Creates Two DataFrames : 
        AppUsage : Google Data Dictionary into Pandas DataFrame
        AppDailyUsage : having App Usage in Daywise
    '''
    global Timezone
    mytimezone = pytz.timezone(' '.join(pytz.country_timezones[Timezone]))
    AppDailyUsage = pd.DataFrame(columns=['App', 'Time'])
    AppName=[];AppUseTime=[]
    for gd in gData:
        timestr = gd['time']
        if '.' not in timestr:
            timestr = timestr.replace('Z','.000Z')
        # Convert Time string into datetime object and then into IST    
        Time = datetime.strptime(timestr, '%Y-%m-%dT%H:%M:%S.%fZ') # time data into datetime object
        utcTime = pytz.UTC.localize(Time) # Make Timezone aware by Adding Time zone UTC
        myTime = utcTime.astimezone(mytimezone) # Change Time object timezone
        
        ## DataFrame 1 used for Sleep Data
        index = str(myTime.year)+'-'+str(myTime.month)+'-'+str(myTime.day)
        if index not in AppDailyUsage.index:
            AppDailyUsage.loc[index, 'App'] = []
            AppDailyUsage.loc[index, 'Time'] = []     
        if index in AppDailyUsage.index:       
            AppDailyUsage.loc[index, 'App'].append(gd['header'])
            AppDailyUsage.loc[index, 'Time'].append(myTime)
    
        ## DataFrame 2 used for App Usage
        # Create Lists as Columns to add into DataFrame
        AppName.append(gd['header']) # AppName List
        AppUseTime.append(myTime) # AppTime List
    print('.',end='')
    AppUsage = pd.DataFrame(AppName, columns=['App'] ) # Add App list while creating Dataframe
    AppUsage['Time'] = AppUseTime # Add Time list
    AppUsage.sort_values(by=['Time'], inplace=True, ascending=False) # Sort by Date in 'Recent First' Order 
    
    return AppUsage, AppDailyUsage



#for i in ApUsgMthChange:
#    print(i,AppUsage.iloc[i, 1])
    
    
'''
def TopApps(n=0):
    """Gives Top n apps otherwise gives Dictionary having Total Counts"""
    ExcludeApps=['com.miui.home' ]
    global gData
    from collections import Counter
    allApps=[ app['header'] for app in gData ]
    APPS = Counter(allApps)
    TopApps=sorted(APPS, key=APPS.get, reverse=True)
    TopApps = [ app for app in TopApps if app not in ExcludeApps ]
    #print(TopApps.most_common(n))
    TopAppS = { app : APPS[app] for app in TopApps }
    
    return  TopApps[:n] if n else TopAppS
'''

def TopApps(AppUsage, n=0, yearly=0):
    """Gives Top n apps otherwise gives Dictionary having Total Counts"""
    
    def getTopApps(AppUsage, n=0):
        global ExcludeApps
        from collections import Counter
        allApps = [ app for app in AppUsage['App'] ] 
        APPS = Counter(allApps)
        TopApps=sorted(APPS, key=APPS.get, reverse=True)
        TopApps = [ app for app in TopApps if app not in ExcludeApps ]
        #TopAppS = { app : APPS[app] for app in TopApps }
        return TopApps[:n] if n else { app : APPS[app] for app in TopApps }

    if yearly:
        year=datetime.now().year
        ApUsgYrChange = [0]
        for i,dt in enumerate(AppUsage['Time']):
            if dt.year != year :
                ApUsgYrChange.append(i)
                year=dt.year
        ApUsgYrChange.append(len(AppUsage))
        TopAppsYearly={}
        for i in range(len(ApUsgYrChange)-1):
            AppUsageYearly = AppUsage[ApUsgYrChange[i]:ApUsgYrChange[i+1]]
            TopAppsYearly[AppUsageYearly['Time'].iloc[-1].year] = getTopApps(AppUsageYearly,n)
        return TopAppsYearly
    else:
        return getTopApps(AppUsage,n)
    



def appUsageData(topApps, AppUsage, Yearly=0, All=0): # Works on AppUsage DataFrame
    '''Gives DataFrame having App Usage calculated yearly of the Top n Apps'''
    global Verbose
    #print(topApps); Yearly=1; All=0
    topApps=[topApps] if isinstance(topApps,str) else ( TopApps(AppUsage,topApps, yearly=1) if not All else TopApps(AppUsage,topApps) ) if isinstance(topApps,int) else topApps # Convert String into List
    print('\n\n   -- Generating App-Usage Report ..',end='') if not Yearly and not All else print(' .',end='')
    year=datetime.now().year; month=datetime.now().month
    # Create Two Lists, Each to store years range index & months range index respectively
    ApUsgYrChange = [0]; ApUsgMthChange = [0]
    for i,dt in enumerate(AppUsage['Time']):
        if dt.year != year :
            ApUsgYrChange.append(i)
            year=dt.year
        if dt.month != month :
            ApUsgMthChange.append(i)
            month=dt.month
    ApUsgYrChange.append(len(AppUsage)); ApUsgMthChange.append(len(AppUsage))

    if Yearly:
        ApUsgMthChange=ApUsgYrChange
    if All:
        YrDuration=str(AppUsage.iloc[-1, 1].year)+'-'+str(AppUsage.iloc[0, 1].year)
        ApUsgMthChange=[0,len(AppUsage)]
    # Segregate whole AppUsage into Months data using month range index
    AppMonthlyUsage={}
    for n in range(len(ApUsgMthChange)-1):
        AppMonthlyData = AppUsage[ApUsgMthChange[n]:ApUsgMthChange[n+1]]

        Year = AppMonthlyData.iloc[0, 1].year
        Month = AppMonthlyData.iloc[0, 1].month
        
        if not Yearly and not All:
            TopAPPS = topApps[Year]
            print(Year,' : ',Month) if Verbose else  print(( '\n      - Year : '+str(Year)+' ' if ApUsgMthChange[n] in ApUsgYrChange else '.' ), end='')
        else:
            if Yearly:
                TopAPPS = topApps[Year]
                if Verbose:
                    print('\n Yr: '+str(Year)+' :: \n')
            if All:
                if Verbose:
                    print('\n '+YrDuration+' :: \n')
                TopAPPS = topApps#[list(topApps.keys())[0]]

        AppUsageDF = pd.DataFrame() # Create New dataframe for Each Month
        # columns=['App','Opened per day', 'Avg. Use: Minutes Daily', 'Total Time(Hrs.)', 'Max (Hr.)']
        AppMonthlyUsage.setdefault(Year, AppUsageDF)
        
        for index,appName in enumerate(TopAPPS):
            if Verbose:
                print('   Finding Usage for : '+appName+' ...')
            AppUsageDict={} ; AppUsageTime={}
            for i in AppMonthlyData.index[::-1]:
                app = AppMonthlyData.loc[i, 'App']
                timeD = AppMonthlyData.loc[i, 'Time']

                key = str(timeD.year)+'-'+str(timeD.month)+'-'+str(timeD.day)
            
                if appName.lower() in app.lower() :
                    AppUsageDict.setdefault(key,[])    
                    AppUsageTime.setdefault(key,[])
                    
                    if i-1 in AppMonthlyData.index:
                        AppUsageDict[key].append( (timeD, AppMonthlyData.loc[i-1,'Time']) )
                        AppUsageTime[key].append(AppMonthlyData.loc[i-1,'Time'].timestamp() - timeD.timestamp())
                    else:
                        AppUsageDict[key].append((0,0))
                        AppUsageTime[key].append(0)
    
            AppTimelist=[] # Stores Duration an App is used throughout the day
            for day in AppUsageTime:
                ttime=sum(AppUsageTime[day])/60   # In Minutes
                #print(day,' : ',ttime)
                AppTimelist.append(ttime)
   
            lenAppUsageTime = 0.00001 if not len(AppUsageTime) else len(AppUsageTime) # ZeroDivisionError

            TotalTimeMin=sum(AppTimelist)
            AppUsageDF.loc[index+1, 'App'] = appName
            AppUsageDF.loc[index+1, 'Opened per day'] = round( sum([len(AppUsageTime[i]) for i in AppUsageTime]) / lenAppUsageTime , 1)
            AppUsageDF.loc[index+1, 'Avg. Use: Minutes Daily'] = round( TotalTimeMin/lenAppUsageTime , 1)
            AppUsageDF.loc[index+1, 'Total Time(Hrs.)'] = round( TotalTimeMin/60 , 1)
            #AppUsageDF.loc[index+1, 'Total Time(Days)'] = TotalTimeHr/24

            AppUsageDF.loc[index+1, 'Maximum Usage (Hr.)'] = str(round( max(AppTimelist)/60 , 1)) + ' ['+list(AppUsageTime.keys())[AppTimelist.index(max(AppTimelist))].replace('-','/')+']' if AppTimelist else 0
            #AppUsageDF.loc[index+1, 'Min (Hr.)'] = str(min(AppTimelist)/60)+' ['+list(AppUsageTime.keys())[AppTimelist.index(min(AppTimelist))]+']'

        ## Create Pandas DataFrame from Dictionary
        if not Yearly and not All: 
           AppUsageDF.insert(loc=0, column='Month', value=[Month]*len(AppUsageDF)) # Add Month Column in the beginning
           AppMonthlyUsage[Year] = pd.concat( [ AppMonthlyUsage[Year], AppUsageDF ], ignore_index=True)
           AppMonthlyUsage[Year].drop_duplicates(inplace = True)
        else:
            if Yearly:
                AppMonthlyUsage[Year] = AppUsageDF
            if All:
                AppMonthlyUsage={}
                AppMonthlyUsage[YrDuration] = AppUsageDF

    return AppMonthlyUsage



def AppUsageAnalysis(AppMonthlyUsage,AppUsage):
    '''Creates Yearly Plots Monthwise and DataFrame Table'''
    global TopAppsN
    #topApps = TopApps(TopAppsN) # To Order columns of Pivot DataFrame 
    topApps = TopApps(AppUsage, TopAppsN, yearly=1)
    
    plotYLabels=['No. of Times App Opened per day', 'Average Daily Use in Minutes', 'Total Time Spent in Hrs.', 'Maximum Time devoted in Hrs.']
    Plots={}
    for year in AppMonthlyUsage:
        for colName,plotYLabel in zip(AppMonthlyUsage[year].columns[2:-1],plotYLabels):
            plot_df = AppMonthlyUsage[year].pivot(index='Month', columns='App', values=colName)
            plot_df=plot_df.loc[:,topApps[year]] # Order Columns in TopApps order
            if 'Max' in colName:
                for col in plot_df.columns:
                    plot_df[col] = plot_df[col].str.split(' \[', expand=True)[0] # Split into columns & Take first column
                plot_df.fillna('0', inplace=True)
                plot_df=plot_df.astype(float)

            plot_df = plot_df.loc[(plot_df!=0).any(axis=1)] # Remove Rows having all Zeroes
            #### Plotting
            ax = plot_df.plot.barh( stacked=True, color=matplotlib.cm.get_cmap('Set3', TopAppsN).colors, figsize=(16,9) )
            
            TotalTime = round(plot_df.T.sum())
            TotalTime.replace(0, 0.000001, inplace=True) # Avoid Division by Zero Error
            allpertg = plot_df.div(TotalTime, 0)*100
            for n in allpertg:
                for i, (cs, ab,  pc, tot) in enumerate(zip(plot_df.iloc[:, :].cumsum(1)[n], plot_df[n], allpertg[n], TotalTime)):
                    ax.text(tot, i, str(tot), va='center') # Total Count
                    if pc:
                        ax.text(cs - ab/2, i, str(round(pc)) +'%('+str(round(ab))+')', va='center', ha='center', rotation=(90 if pc < 8 else 0), fontsize = 11, fontweight='bold' ) # % for Each bar

            ax.set_xlabel(plotYLabel, labelpad=10, fontweight='bold', fontsize = 14)
            
            ax.invert_yaxis()
            ax.set_ylabel('Month', labelpad=10, fontweight='bold', fontsize = 16)
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_yticklabels([calendar.month_name[int(yl)] for yl in ylabels])
            #plt.xticks(fontsize=12)#, rotation=90)
            #plt.yticks(np.arange(0, 14, 1), fontsize=14)
            #ax.set_yticks(np.arange(0, 14, 1))
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            
            ax.set_title('Apps '+colName+' in '+str(year), fontweight='bold', fontsize = 16)
            
            Figname=('App-'+str(year)+'-'+str(plotYLabel)+'.png').replace(' ','_')
            Plots.setdefault(str(year), [])  # Add Plot names
            Plots[str(year)].append(str(year)+'/'+Figname)
            #plt.tight_layout()
            #ax.figure.set_size_inches(16, 9)
            FilePath=ReportName+'App-Usage'+'/'+str(year)+'/'
            makeDir(FilePath)
            ax.figure.savefig(FilePath+Figname, dpi = 100, bbox_inches='tight')     
            #ax.figure.show()


    AppYearlyUsage = appUsageData(TopAppsN, AppUsage, Yearly=1) # Get Table of Yearly Data   
    AppAllTimeUsage = appUsageData(TopAppsN, AppUsage, All=1) # Get Table for Entire Period
    AppAllTimeUsage.update(AppYearlyUsage)
    
    return AppAllTimeUsage, Plots # Returns DataFrame Table & Plots name for Report generation
 



def SleepData2(AppDailyUsage):
    '''Generates Sleep data i.e. bedtime & wake-up time from the Second DataFrame'''
    Sleep={}
    global timed
    InvalidSleepTime = 13
    for date in AppDailyUsage.index :
        beforeSleep=[]; afterSleep=[]    
        for dt in AppDailyUsage.loc[date, 'Time'] :
            LateSleepTime = datetime(dt.year, dt.month, dt.day, timed['sleep'][0], timed['sleep'][1])
            LateWakeupTime = LateSleepTime.replace(hour=timed['wake'][0], minute=timed['wake'][1], second=0, microsecond=0)             
            
            DayTime = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
            if DayTime < LateSleepTime :
                beforeSleep.append(dt)
            elif DayTime > LateWakeupTime:
                afterSleep.append(dt)

        #print(beforeSleep, afterSleep)        
        if afterSleep :
            if not beforeSleep: # If no Time of Sleep after 12 am
                pdt = dt - timedelta(days=1) # Go to Previous day
                pindex = str(pdt.year)+'-'+str(pdt.month)+'-'+str(pdt.day)
    
                if pindex in AppDailyUsage.index:
                    BeforeSleep = AppDailyUsage.loc[pindex, 'Time'][0] # Last activity of the Previous day
                else: # Beginning of the Current day
                    BeforeSleep = afterSleep[0].replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                BeforeSleep = beforeSleep[0]
            #BeforeSleep = afterSleep[0].replace(hour=0, minute=0, second=0, microsecond=0) if not beforeSleep else beforeSleep[0]
            sleeptime = (afterSleep[-1].timestamp() - BeforeSleep.timestamp()) / 3600
            if sleeptime < InvalidSleepTime:
                Sleep[date] = [ BeforeSleep, afterSleep[-1], sleeptime ]

    return Sleep



def SleepData(AppDailyUsage):
    '''Generates Sleep data from the Second DataFrame'''
    Sleep={}
    global timed
    SleepLimit = 13
    SleepTimeRange={ 'bed':[19,0], 'wake':[14,0], 'latewake': [16,0] }
    ShortSleepRange = { 'bed':[0,0], 'wake':[11,0] }
    ValidSleep = 2.5
    #for i, day in enumerate(AppDailyUsage[::-1].index[:2]) :
    for i in range(len(AppDailyUsage)-1) : # In Recent order
        today  = AppDailyUsage.iloc[i, 1]
        yesterday=AppDailyUsage.iloc[i+1, 1]
        #print(yesterday,' : ',today, '\n\n')
        todayStart = datetime(yesterday[0].year, yesterday[0].month, yesterday[0].day, SleepTimeRange['bed'][0], SleepTimeRange['bed'][1])
        todayEnd = datetime(today[0].year, today[0].month, today[0].day, SleepTimeRange['wake'][0], SleepTimeRange['wake'][1])
        todayDT=str(todayEnd.year)+'-'+str(todayEnd.month)+'-'+str(todayEnd.day) # Date as Key

        ## Check if Previous day Exists
        pdt = todayEnd - timedelta(days=1) # Go to Previous day
        pdate = str(pdt.year)+'-'+str(pdt.month)+'-'+str(pdt.day)
        if pdate not in AppDailyUsage.index: # If No Previous Day
            todayStart = datetime(today[0].year, today[0].month, today[0].day, 0, 0)
            yesterday=today
            
            
        Today = sorted([ dt for dt in yesterday if datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute) > todayStart ] + [ dt  for dt in today if todayEnd > datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute) ], reverse=True)
        if len(Today)>1:
            TimeDur=[ (Today[t] - Today[t+1]).seconds for t in range(len(Today)-1) ]
            MaxTime = max(TimeDur) ;  MaxIndex = TimeDur.index(MaxTime)
            SleepTime=round(MaxTime/3600,1)
            
            wt=datetime(Today[MaxIndex].year, Today[MaxIndex].month, Today[MaxIndex].day, Today[MaxIndex].hour, Today[MaxIndex].minute)

            if wt < datetime(today[0].year,today[0].month, today[0].day, SleepTimeRange['latewake'][0], SleepTimeRange['latewake'][1]) : # If wake time before 4pm
                if ValidSleep <= SleepTime <= SleepLimit: # If sleep b/w 2.5 & 13
                    Sleep[todayDT] = [ Today[MaxIndex+1], Today[MaxIndex], SleepTime ]
                elif 0 < SleepTime < ValidSleep: # For Sleep less than 2.5
                    d1 = datetime(today[0].year,today[0].month, today[0].day, ShortSleepRange['bed'][0], ShortSleepRange['bed'][1])
                    #d2 = datetime(today[0].year,today[0].month, today[0].day, ShortSleepRange['wake'][0], ShortSleepRange['wake'][1])
                    d2 = d1.replace(hour=ShortSleepRange['wake'][0], minute=ShortSleepRange['wake'][1], second=0, microsecond=0)
                    td=datetime(Today[MaxIndex+1].year, Today[MaxIndex+1].month, Today[MaxIndex+1].day, Today[MaxIndex+1].hour, Today[MaxIndex+1].minute)
                    if  d1 <= td <= d2:
                        Sleep[todayDT] = [ Today[MaxIndex+1], Today[MaxIndex], SleepTime ]
                    else:
                        pass
        
    return Sleep






def SleepAnalysis(Sleep):
    '''Segregate Sleep Data Yearwise & Monthwise, Plots them using barplot & scatter function respectively and Generates Yearly & Monthly Sleep Stats DataFrame'''
    global Verbose
    year=datetime.now().year; month=datetime.now().month
    YrChange = []; MthChange = []
    for i,dt in enumerate(list(Sleep.keys())):
        dty,dtm,_ = dt.split('-')
        if dty != year :
            YrChange.append(i)
            year=dty
        if dtm != month :
            MthChange.append(i)
            month=dtm
    YrChange.append(len(Sleep)); MthChange.append(len(Sleep))

    barplots={'yearly':[], 'monthly':[]}
    scatterplots={'yearly':[], 'monthly':[]}
    SleepYearlyMeanDF=pd.DataFrame()
    SleepMonthlyMeanDF={}
    print('\n   -- Generating Sleep Report ',end='')
    ## Split Data Yearly
    for n in range(len(YrChange)-1):
        xdata=[ dt for dt in Sleep ][YrChange[n]:YrChange[n+1]]
        if len(xdata)>1:
            # bar plot
            ydata=[ Sleep[dt][-1] for dt in xdata ]
            
            Year = datetime.strptime(xdata[0], '%Y-%m-%d').strftime('%Y')
            Title=Year
            if Verbose:
                print(' Yearly :',Title,' :: Bar Plotting - ',end='')
            plotName=str(Year)+'-ST.png'
            sleepInfo = barplot(xdata, ydata, plotName, yearly=1)
            barplots['yearly'].append(str(Year)+'/'+plotName)
    
        
            # Scatter Plot
            xsleeptime =  [ (Sleep[dt][0].hour if Sleep[dt][0].hour < 12 else Sleep[dt][0].hour-24) + (Sleep[dt][0].minute/60 + Sleep[dt][0].second/3600) for dt in xdata ][::-1] # Convert Time into Hr for stats calculation & plotting
            ywakeuptime = [Sleep[dt][1].hour + Sleep[dt][1].minute/60 + Sleep[dt][1].second/3600 for dt in xdata ][::-1]
            plotName=Year+'-SR.png'
            print('Scatter Plotting') if Verbose else print('.',end='')
            bedtime,waketime = scatterplot(xsleeptime, ywakeuptime, xdata, Title,plotName, yearly=1)
            scatterplots['yearly'].append(str(Year)+'/'+plotName)
            YearIndex=Year+' ('+str(len(xsleeptime))+')'
            SleepYearlyMeanDF.loc[YearIndex, list(sleepInfo.keys())[0]] = sleepInfo[list(sleepInfo.keys())[0]] # Mean Sleep-Time
            SleepYearlyMeanDF.loc[YearIndex, 'Mean Bed-Time'] = bedtime
            SleepYearlyMeanDF.loc[YearIndex, 'Mean WakeUp-Time'] = waketime
            SleepYearlyMeanDF.loc[YearIndex, list(sleepInfo.keys())[1]] = sleepInfo[list(sleepInfo.keys())[1]] # Total Time Spent Sleeping
            SleepYearlyMeanDF.loc[YearIndex, list(sleepInfo.keys())[2]] = sleepInfo[list(sleepInfo.keys())[2]] # Sleep-Deficit/Sleep-Debt
            SleepYearlyMeanDF.loc[YearIndex, list(sleepInfo.keys())[3]] = sleepInfo[list(sleepInfo.keys())[3]] # Total Sleeping(Hrs)
    
    
    ## Split Data Monthly
    for n in range(len(MthChange)-1):
        xdata=[ dt for dt in Sleep ][MthChange[n]:MthChange[n+1]][::-1]
        if len(xdata)>1:
            # bar plot
            ydata=[Sleep[dt][-1] for dt in xdata ][::-1]
            
            Year = datetime.strptime(xdata[0], '%Y-%m-%d').strftime('%Y')
            Month = datetime.strptime(xdata[0], '%Y-%m-%d').strftime('%B')
            Title=Month+', '+Year
            if Verbose:
                print(' Monthly :',Title,' :: Bar Plotting - ',end='')
            plotName=Year+'-'+str(datetime.strptime(xdata[0], '%Y-%m-%d').month)+'-ST.png'
            sleepInfo = barplot(xdata, ydata, plotName)
            barplots['monthly'].append(str(Year)+'/'+plotName) # Add Plot names
     
        
            # Scatter Plot
            #xsleeptime =  [ (Sleep[dt][0].hour if Sleep[dt][0].hour < 12 else Sleep[dt][0].hour-24)  + Sleep[dt][0].minute/60 + Sleep[dt][0].second/3600 for dt in xdata ][::-1] # Convert Time into Hr for stats calculation & plotting
            xsleeptime =  [ (Sleep[dt][0].hour if Sleep[dt][0].hour < 12 else Sleep[dt][0].hour-24) + (Sleep[dt][0].minute/60 + Sleep[dt][0].second/3600) for dt in xdata ][::-1] # Convert Time into Hr for stats calculation & plotting
            ywakeuptime = [ Sleep[dt][1].hour + Sleep[dt][1].minute/60 + Sleep[dt][1].second/3600 for dt in xdata ][::-1]
            
            print('Scatter Plotting') if Verbose else  print(( '\n      - Year : '+str(Year)+' ' if MthChange[n] in YrChange else '.' ), end='')
            plotName=Year+'-'+str(datetime.strptime(xdata[0], '%Y-%m-%d').month)+'-SR.png'
            bedtime,waketime = scatterplot(xsleeptime, ywakeuptime, xdata, Title,plotName)
            scatterplots['monthly'].append(str(Year)+'/'+plotName)

            SleepMonthlyMeanDF.setdefault(Year, pd.DataFrame())
            MonthIndex=Month+' ('+str(len(xsleeptime))+')'
            SleepMonthlyMeanDF[Year].loc[MonthIndex, list(sleepInfo.keys())[0]] = sleepInfo[list(sleepInfo.keys())[0]] # Mean Sleep-Time
            SleepMonthlyMeanDF[Year].loc[MonthIndex, 'Mean Bed-Time'] = bedtime
            SleepMonthlyMeanDF[Year].loc[MonthIndex, 'Mean WakeUp-Time'] = waketime
            SleepMonthlyMeanDF[Year].loc[MonthIndex, list(sleepInfo.keys())[1]] = sleepInfo[list(sleepInfo.keys())[1]] # Total Time Spent Sleeping
            SleepMonthlyMeanDF[Year].loc[MonthIndex, list(sleepInfo.keys())[2]] = sleepInfo[list(sleepInfo.keys())[2]] # Sleep-Deficit/Sleep-Debt
            SleepMonthlyMeanDF[Year].loc[MonthIndex, list(sleepInfo.keys())[3]] = sleepInfo[list(sleepInfo.keys())[3]] # Total Sleeping(Hrs)
             
    SleepMonthlyMeanDF = { year : SleepMonthlyMeanDF[year].iloc[::-1] for year in  SleepMonthlyMeanDF }
            
    return  SleepYearlyMeanDF, SleepMonthlyMeanDF, barplots, scatterplots




def barplot(xdata, ydata, plotName, yearly=0):
    
    from matplotlib import pyplot as plt
    import matplotlib.transforms as transforms
    from mpl_toolkits.axes_grid.inset_locator import inset_axes # To add another plot
    #import matplotlib.patches as mpatches
    
    import numpy as np
    from datetime import datetime
    global IdlSleepT, ReportName
    meanSleep=round(np.mean(ydata),1)
    #plt.xkcd()
    #plt.style.use('fivethirtyeight')

    width = 0.8  # the width of the bars
    '''
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    '''

    fig, ax = plt.subplots()

    #col = ['g'  if i>6  else 'y' if i>4 else 'r'   for i in ydata]

    #barColor = { 'green': ['g','#64DD17'], 'red': ['r', '#E53935'], 'yellow': ['y','#FFEB3B']  }
    bGreen,bYellow,bRed = 'g','y','r'
    col = [bGreen  if i>=IdlSleepT  else bYellow if i>(IdlSleepT-2.5) else bRed  for i in ydata]
    ## Year & Month for Title
    Year = datetime.strptime(xdata[0], '%Y-%m-%d').strftime('%Y')
    Month = datetime.strptime(xdata[0], '%Y-%m-%d').strftime('%B')
    xlabel='Month'
    BoxInfoLabel=Title=Year
    if yearly: # To plot yearly data
        from matplotlib import dates as mdates, ticker
        
        xdates=[datetime.strptime(dt, '%Y-%m-%d') for dt in xdata]  # Convert xaxis data into dates
        #plt.clf() # Clear Current Fig.
    
        ax.bar(xdates, ydata , width, color=col, align='center')
        #ax.plot_date(xdates, ydata, linestyle='solid', marker='None', color='g', label='Sleeping Hrs.')
        #ax.stem(xdates, ydata)        
        #ax.hlines(y=ydata, xmin=0, xmax=ordered_df['values'], color='skyblue')

        #Set xticks at Every Month
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        #Set Ticks Format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


    else:
        #width= 0.7
        ### Plotting 
        bGreen,bYellow,bRed = '#64DD17','#FFEB3B','#E53935'
        # Create colour list having colour for each bar
        col = [bGreen  if round(i,1)>IdlSleepT  else bYellow if round(i,1)>(IdlSleepT-2.5) else bRed for i in ydata]

        bars = ax.bar( [ dt.split('-')[-1] for dt in xdata], ydata,  width, color=col, align='center')
        ## Value on Top of Each Bar
        for i,rect in enumerate(bars):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, str(round(ydata[i],1)), ha='center', va='bottom', fontsize = 10)        
        
        xlabel='Day of '+Month
        BoxInfoLabel=Month
        Title=Month+', '+Year


    plt.axhline( IdlSleepT, color='#00C853', linestyle='dashed', linewidth=2) # Ideal Line
    plt.axhline( meanSleep, color='#8E24AA', linestyle='dashed', linewidth=3) # Data Mean Line
    ## Mark Data on Y Axes
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0,meanSleep, 'Mean:   \n'+str(meanSleep), color="#8E24AA", fontsize=12, fontweight='bold', transform=trans,  ha="right", va="center")
    ax.text(0,IdlSleepT, 'Ideal:   ', color="#00C853", fontsize=12, fontweight='bold', transform=trans,  ha="right", va="center")

    #textstr = 'In '+BoxInfoLabel+' :: \n Total Time Spent Sleeping : '+str(round((sum(ydata)/(len(ydata)*24))*100,1))+' %  [Ideal: '+str(round(((len(ydata)*IdlSleepT)/(len(ydata)*24))*100,1))+' %]\n Mean Sleep Time Hrs. : '+str(meanSleep)+' [Ideal: '+str(IdlSleepT)+']'+'\n Total Sleeping Hrs. : '+str(round(sum(ydata),1))+'/'+str(round(len(ydata)*24,1))
    ## Adding Info Textbox    
    textstr = 'In '+BoxInfoLabel+' :: '+'\n Mean Sleep Time Hrs. : '+str(meanSleep)+'  (Ideal: '+str(IdlSleepT)+')'+'\n\n Total Time Spent Sleeping : '+str(round((sum(ydata)/(len(ydata)*24))*100,1))+' %  (Ideal: '+str(round(((len(ydata)*IdlSleepT)/(len(ydata)*24))*100,1))+' %)'+'\n Sleep Deficit / Sleep Debt : '+str(round(33.3333333 - (sum(ydata)/(len(ydata)*24)*100),1))+' % \n\n Total Sleeping Hrs. : '+str(round(sum(ydata),1))+' / '+str(round(len(ydata)*24,1))+' (Ideal: '+str(round((len(ydata)*IdlSleepT),1))+')'
    # Text Box in upper Right in axes coords
    props = dict(boxstyle='round', facecolor='#8C9EFF', alpha=0.5)
    ax.text( 0.67, 0.98, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    
    ## Pie Chart
    left, bottom, width, height = [0.16, 0.69, 0.17, 0.17] # Loation & Size of the Plot
    ax2 = fig.add_axes([left, bottom, width, height])
    #ax2.plot(range(6)[::-1], color='green')
  
    sliceData = [col.count(bGreen), col.count(bRed), col.count(bYellow)] # Data for Each Slice
    sliceLabel = [ 'Sleep > '+str(IdlSleepT)+' Hrs.', 'Sleep < '+str(IdlSleepT-2.5)+' Hrs.', 'Sleep b/w '+str(IdlSleepT-2.5)+' & '+str(IdlSleepT)+' Hrs.'] # Label for Each Slice
    #cols = ['#CCFF90', '#F4FF81', '#FF9E80']
    cols = ['#B2FF59', '#FF6E40', '#FFFF8D']

    ax2.pie(sliceData, labels=sliceLabel, colors=cols, startangle=90, shadow= True, explode=(0,0.1,0), autopct=lambda pcent: '{:.2f}% \n ({:.0f})'.format(pcent,(pcent/100)*len(col)) , wedgeprops={'alpha':0.8}, textprops=dict(  weight='bold', fontsize=9) ) # color='w',
    #'%1.1f%%
    ax2.set_title(' Total Days: '+str(len(col)), fontdict={'fontsize':9, 'fontweight':'bold'}, pad=0.1)  # Set Title
  
    
    #ax.set_ylim(0,75)
    #plt.xticks(np.arange(0, 225+10 , 20))
    #ax.set_yticks(x + width*0.90 )
    #ax.set_yticklabels([   for i in   ], fontsize = 19)
    '''
    ## Add Custom Legends
    legend_dict = { 'Sleep > '+str(IdlSleepT)+' Hrs. : '+str(round((col.count(bGreen)/len(col)*100),2))+'%' : bGreen, 'Sleep < '+str(IdlSleepT-2.5)+' Hrs. : '+str(round((col.count(bRed)/len(col)*100),2))+'%' : bRed, 'Sleep b/w '+str(IdlSleepT-2.5)+' & '+str(IdlSleepT)+' Hrs. : '+str(round((col.count(bYellow)/len(col)*100),2))+'%' : bYellow }
    patchList = [ mpatches.Patch(color=legend_dict[key], label=key) for key in legend_dict ]
    ax.legend( loc=2, handles=patchList, prop={'size': 15}) # 
    '''

    ax.set_xlabel(xlabel, fontweight='bold', fontsize = 18)
    ax.set_ylabel('Hours',labelpad=40, fontweight='bold', fontsize = 18)
    #plt.xticks(fontsize=12)#, rotation=90)
    #plt.yticks(np.arange(0, 14, 1), fontsize=14)
    ax.set_yticks(np.arange(0, 14, 1))
    
    ax.set_title('Sleep Time in '+Title, fontweight='bold', fontsize = 18)
    
    #ax.legend(prop={'size': 15})

    Figname=str(plotName)
    #plt.tight_layout()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 9)
    #plt.savefig(Figname, )
    FilePath=ReportName+'Sleep-Routine'+'/'+Year+'/'
    makeDir(FilePath) 
    plt.savefig(FilePath+Figname, dpi = 200, bbox_inches='tight')
    #plt.show()
    sleepInfo = { 'Mean\nSleep-Time(Hrs)' : str(meanSleep) , 'Total Time Spent Sleeping (Ideal:33.3%)': str(round((sum(ydata)/(len(ydata)*24))*100,1))+'%', 'Sleep-Deficit/\nSleep-Debt': str(round(33.3333333 - (sum(ydata)/(len(ydata)*24)*100),1))+'%', 'Total Sleeping(Hrs)': str(round(sum(ydata),1))+'/'+str(round(len(ydata)*24,1))+' (Ideal:'+str(round((len(ydata)*IdlSleepT),1))+')' }

    return sleepInfo





def scatterplot(xsleeptime,ywakeuptime,xdata, Title, plotName, yearly=0, markpoints=0):
    from matplotlib import pyplot as plt
    import matplotlib.transforms as transforms
    import matplotlib.font_manager as Fontmanager
    
    import math
    global IdlSleepT, ReportName

    
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    
    fig, ax = plt.subplots()
    
    idlBT=0; idlWT=8
    
    markerscaling = 10 if not yearly else 6

    #cmap = plt.get_cmap('plasma')
    cmap='viridis_r'
    cmap='jet_r'
    cmap='Dark2_r'
    #cmap='plasma_r'

    
    sleeptimes=[ abs(wt-bt) for bt,wt in zip(xsleeptime,ywakeuptime) ]
    
   # xsleeptime = [ t-24 if 12<t<24 else t  for t in xsleeptime] # Change Sleeptime for previous day

    #ax.scatter(xsleeptime, ywakeuptime, color=colors)
    points=ax.scatter(xsleeptime, ywakeuptime, c=sleeptimes, cmap=cmap, s=sleeptimes, linewidth=markerscaling, alpha=1)
 
    if markpoints:
        pfmt = '%m/%d' if len(xdata)>31 else '%d'
        xdays=[datetime.strptime(d, '%Y-%m-%d').strftime(pfmt) for d in xdata]
        for x,y, d in zip(xsleeptime, ywakeuptime, xdays):
            ax.text(x+0.07, y, d, va='center', fontsize = 7, fontweight='bold' )
            #ax.annotate( d, (x, y), va='center', ha='center', fontsize = 5, fontweight='bold' )
    ## Set custom colorbar range
    #cbardlim,cbarulim = round(math.floor(min(sleeptimes)/markerscaling),1),round(math.ceil(max(sleeptimes)/markerscaling),1)
    #boundaries=np.linspace(cbardlim,cbarulim,cbarulim-cbardlim)) 
    
    #plt.colorbar(points)#.set_label(label='Sleeping Hrs.', size=13,weight='bold')
    
    cb = plt.colorbar(points, label='Sleeping Hrs.')
    axcb = cb.ax # Get axis of the colorbar plot
    ylabeltext = axcb.yaxis.label # Get label of the desired axis
    font = Fontmanager.FontProperties(family='times new roman', weight='bold', size=14) # Create Matplotlib Text object # style='italic', 'normal'
    ylabeltext.set_font_properties(font) # Set it to colorbar axis

    #plt.axhline( IdlSleepT, color='#00C853', linestyle='dashed', linewidth=2) # Ideal Line
    btmean=round(np.mean(xsleeptime),1) # Mean Bed Time
    wtmean=round(np.mean(ywakeuptime),1) # Mean Wake Up Time

    def addMeanTimeFmt(timelist, floatTime, addchar=''): # Replace Num. Time into Time in a List
         tpos,tampm=[(i,tstr) for i,tstr in enumerate(timelist) if '.' in tstr][0]
         ampm = 'am' if 'am' in tampm else 'pm'
         #TimeFmt=str(floatTime).split('.')[0]+':'+str(int(str(floatTime).split('.')[1])*60)[:2]+' '+ampm # Converts Float Time into Time Format
         minut,hr = math.modf(floatTime)
         
         TimeFmt=( ( str(math.ceil(hr) if math.ceil(hr) else 12) +':'+f"{str(int(round(minut*60,1))):0>2}")  if floatTime >= 0 else ( str(math.floor(floatTime)+12)+':'+f"{str(int(round((1-abs(minut))*60,1))):0>2}" ) ) +' '+ampm # Converts Float Time into Time Format  ### .replace('0','12')
         
         timelist[tpos] = TimeFmt+tampm.split(ampm)[-1] # Replace Numerical Time with Time format
         if addchar:
             timelist[tpos] = timelist[tpos]+addchar if ' ' in addchar else addchar+timelist[tpos]
         return timelist,TimeFmt

    
    #plt.axvline( stmean, color='#8E24AA', linestyle='dashed', linewidth=1) # Mean x Line
    #plt.axhline( wtmean, color='#8E24AA', linestyle='dashed', linewidth=1) # Mean y Line
    #ax.text(0,meanSleep, 'Mean:   \n'+str(meanSleep), color="#00C853", fontweight='bold', transform=trans,  ha="right", va="center")
    '''
    xtranstxt = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
    axt = ax.text(btmean, math.floor(min_ylim), btime, color="#8E24AA", fontsize = 17, fontweight='bold', transform=xtranstxt,  ha="right", va="center")
    axt.set_alpha(.5)
    
    ytranstxt = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ayt = ax.text(0,wtmean, wtime, color="#8E24AA", fontsize = 17, fontweight='bold', transform=ytranstxt,  ha="right", va="center")
    ayt.set_alpha(.5)    '''
    # μ

    #plt.xticks(np.arange(16, 14, 1), fontsize=12)#, rotation=90)
    #fig.canvas.draw()
    #xlabels = [item.get_text() for item in ax.get_xticklabels()]
    #ylabels = [item.get_text() for item in ax.get_yticklabels()]

    # Insert Mean Values on x & y axes
    xvalueL,xvalueH = math.floor(min(xsleeptime)),math.ceil(max(xsleeptime))
    xvalues=[ i for i in range(xvalueL,xvalueH+1)]
    xvalues.insert([i for i,v in enumerate(xvalues) if v <=btmean<= v+1][0]+1, btmean) # Insert Mean value as Tick
    yvalueL,yvalueH = math.floor(min(ywakeuptime)),math.ceil(max(ywakeuptime))
    yvalues=[ i for i in range(yvalueL,yvalueH+1)]
    yvalues.insert([i for i,v in enumerate(yvalues) if v <=wtmean<= v+1][0]+1, wtmean) # Insert Mean value as Tick
    ax.set_xticks(xvalues)
    ax.set_yticks(yvalues)

    # Change Label names
    xtimedict={ str(i):str(i)+'am' if float(i) >= 0 else str(i)+'pm'  for i in xvalues }
    xtimedict.update({'-5':'7pm', '-4':'8pm', '-3':'9pm', '-2':'10pm', '-1':'11pm', '0':'12am'}) # Adding am,pm to x Labels
    #xtimedict.update({ str(i):str(i)+'am' for i in xvalues if float(i) > 0 }) # Adding am,pm to x Labels
    xticklabels=[ xtimedict[str(xl)] for xl in xvalues ]
    yticklabels=[str(i)+'am' if float(i) < 12 else str(i)+'pm' if float(i)==12 else str(i-12)+'pm'   for i in yvalues]
    #yticklabels=[str(i)+'am' if int(i) < 12 else str(i)+'pm' if int(i)==12 else str(i-12)+'pm'   for i in yvalues]

    xticklabels, xMeanTime = addMeanTimeFmt(xticklabels, btmean, addchar='\n')
    yticklabels, yMeanTime = addMeanTimeFmt(yticklabels, wtmean, addchar='      ')
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    #ax.set_yticklabels([ float2TimeFmt(wtmean)+'      ' if isinstance(i,float) else str(i)+'am'  if int(i) < 12 else str(i)+'pm' if int(i)==12 else str(i-12)+'pm'   for i in yvalues]) # Adding am,pm to y Labels
    #ax.set_yticklabels([ wtime+'      ' if '.' in str(i) else str(i)+'am'  if int(i) < 12 else str(i)+'pm' if int(i)==12 else str(i-12)+'pm'   for i in yvalues]) # Adding am,pm to y Labels

    #plt.yticks(fontsize=14)

    ## Info Text Box
    textstr = 'Mean Sleep Time Hrs. : '+str(round(np.mean(sleeptimes),1))+'  (Ideal: '+str(IdlSleepT)+')'+'\n\n Mean Bed Time:  '+xMeanTime+' \n Mean Wake-Up Time:  '+yMeanTime
    # Text Box in upper Right in axes coords
    props = dict(boxstyle='round', facecolor='#8C9EFF', alpha=0.3)
    ax.text( 0.62, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    
    ax.set_xlabel('Bed Time', labelpad=20, fontweight='bold', fontsize = 16)
    ax.set_ylabel('Wake Up Time',labelpad=20, fontweight='bold', fontsize = 16)
    ax.set_title('Sleep Routine in '+Title,fontweight='bold', fontsize = 16)
    
    # if yearly:
    #     plt.xlim(-3,9)
    #     plt.ylim(2,15)

    minxlim=plt.xlim()[0] 
    minylim=plt.ylim()[0] 
    ax.plot( [btmean,minxlim],[minylim,wtmean], '*', markersize=17, color="#8E24AA", zorder=10, clip_on=False)
    ax.set_xlim(left=minxlim)
    ax.set_ylim(bottom=minylim)
    '''
    ax.annotate( str(btime), (btmean,minylim), xytext=(0.01,0.01), textcoords=('axes fraction','axes fraction'), fontweight='bold', fontsize = 16,color="#8E24AA", ha='center', va='center')
    ax.annotate( str(wtime), (minxlim,wtmean), xycoords='data', fontweight='bold', fontsize = 16,color="#8E24AA",  ha='right', va='center')
    #xytext=(2, 3),
    # , shrink=0.05
    '''
    
    Figname=str(plotName)
    #plt.tight_layout()
    figure = plt.gcf() # get current figure
    #figure.set_size_inches(10, 7)
    figure.set_size_inches(16, 9)
    Year = datetime.strptime(xdata[0], '%Y-%m-%d').strftime('%Y')
    FilePath=ReportName+'Sleep-Routine'+'/'+Year+'/'
    makeDir(FilePath) 
    plt.savefig(FilePath+Figname, dpi = 100, bbox_inches='tight')
    #plt.show()

    return xMeanTime, yMeanTime



htmlCSS='''<!DOCTYPE html><html><head><style>
div.gallery {
  margin: 15px;
  border: 2px solid #777;
  float: left;
  }

div.gallery:hover {
  border: 4px solid #ccc;
}
div.gallery img {
  width: 100%;
  height: auto;
}
div.desc {
  padding: 10px;
  text-align: center;
  font-size: 30px;
  font-weight: bold;
  text-decoration-line: underline;
  text-decoration-style: solid;
  margin-bottom: 5px;
}
	.columnpadding {
	  float: left;
	  width:  8%;
	  padding: 5px; }

	.imgcolumn {
	  float: left;
	  width:  80%;
	  padding: 5px; }
	  
	.imgcol2 {
	  float: left;
	  width:  48%;
	  padding: 5px; }

	.row::after {
	  content: "";
	  clear: both;
	  display: table; }  
	  
</style></head><body>

<div class="row">
'''       

def getb64Image(imagePath):
    global InlineImg, ReportName
    if InlineImg:
        with open(ReportName+imagePath, "rb") as img_file:
            dataURI='data:image/png;charset=utf-8;base64, '
            return dataURI + base64.b64encode(img_file.read()).decode('utf-8')
    else:
        return imagePath


def addImageColumnHTML(Plots, plotpath='', title=''):
    '''Generates HTML having Stacked Images from List of Plot names
    '''
    global htmlCSS

    HTML=htmlCSS
    if title:
        HTML+='<hr><div class="desc">'+str(title)+'</div>'

    HTML+='<div class="columnpadding"></div><div class="imgcolumn">' # columnpadding for spacing in Left Column
    
    if isinstance(Plots, dict):
        for Plot in Plots:
            HTML+='<hr><div class="desc">In '+str(Plot)+'</div>'
            if isinstance(Plots[Plot], list):
                for plot in Plots[Plot]:

                    HTML+='''<div class="gallery">
                    <img src="'''+getb64Image(plotpath+plot)+'''"  alt="'''+str(plot).split('.')[0]+'''" width="1200" height="800"></div>'''
            else:
                HTML+='''<div class="gallery">
                    <img src="'''+getb64Image(plotpath+Plot)+'''"  alt="'''+str(Plot).split('.')[0]+'''" width="1200" height="800"></div>'''                
    else:
        for plot in Plots:
            HTML+='''<div class="gallery">
                    <img src="'''+getb64Image(plotpath+plot)+'''"  alt="'''+str(plot).split('.')[0]+'''" width="1200" height="800"></div>'''
    HTML+='</div></div>' # End First Row & Column

    return HTML


      
def addTableHTML(TableData,TableHeading='', HeadingFmt=''):
    CSS='''<style>
    #custom {
      font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
      border-collapse: collapse;
      margin-left: auto; margin-right: auto;
      width: 65%; }
    #custom td, #custom th {
      border: 2px solid #ddd;
      padding: 2px; }
    #custom tr:nth-child(even){background-color: #f2f2f2;}
    #custom th {
      background-color: #5b9bd5;
      color: white; }
    </style>'''
    tableHTML=CSS
    if isinstance(TableData,dict): # If DataFrame is in a Dictionary
        for i, tableHeading in enumerate(TableData):
            TABHeading = str(tableHeading) if not isinstance(TableHeading,list) else str(TableHeading[i]) # If List of Titles is passed, its elements are set as Title of table 
            TABHeading = HeadingFmt.replace('_TITLE_',TABHeading) if '_TITLE_' in HeadingFmt else TABHeading
            tableHTML+=TableData[tableHeading].to_html()+'<br><hr><br><br>' # Add Table Data
            tableHTML=tableHTML.replace('"dataframe"','"Dataframe" id="custom" style="text-align:center"><caption><h2><b><u>'+str(TABHeading)+'</u></b></h2></caption') # Add CSS & Heading

    else: # if Just Pandas DataFrame and a Title is given
        tableHTML=TableData.to_html().replace('"dataframe"','"dataframe" id="custom" style="text-align:center"><caption><h2><b><u>'+str(TableHeading)+'</u></b></h2></caption')
    tableHTML=tableHTML.replace('<tr style="text-align: right;">','<tr style="text-align: center;">')
        
    return  tableHTML
    #with open('ok123.html','w') as file:
    #    file.write(tableHTML)


def generateAppUsageHTML(AppAllTimeUsage, Plots):
    global TopAppsN, ReportName
    HTML=''
    Title=str(TopAppsN)+' Top-Apps Average Usage during '+list(AppAllTimeUsage.keys())[0]
    
    HTML+=addTableHTML(AppAllTimeUsage,TableHeading=Title)
    HTML+=addImageColumnHTML(Plots, plotpath='App-Usage'+'/', title='Average App Usage ')
    
    AppUsageFile=ReportName+'App-Usage Report  '+list(AppAllTimeUsage.keys())[0]+'.html'
    with open(AppUsageFile,'w') as file:
        file.write(HTML)
    print("\n   -- Opening '"+AppUsageFile+'" ..')
    os.system('start CHROME.EXE "'+AppUsageFile+'"')
    return AppUsageFile

    
def generateSleepReportHTML(SleepYearlyMeanDF,SleepMonthlyMeanDF, barplots, scatterplots):
    global htmlCSS, ReportName
    FilePath='Sleep-Routine'+'/'

    def addDesc(txt):
        return '<hr><div class="desc">'+str(txt)+'</div>'
    HTML=htmlCSS
    
    Period=barplots['yearly'][-1].split('/')[0]+'-'+barplots['yearly'][0].split('/')[0]
    HTML+=addTableHTML(SleepYearlyMeanDF,TableHeading='Sleep Taken during '+Period)+addTableHTML(SleepMonthlyMeanDF, HeadingFmt='In _TITLE_')


    ### Yearly Plots :
    HTML+=addDesc('Sleep Time during '+Period)+'''<div class="columnpadding"></div>
    <div class="imgcolumn">'''
    for plot in barplots['yearly']:
        HTML+='''<div class="gallery">
                <img src="'''+getb64Image(FilePath+plot)+'''"  alt="Sleep Time in '''+plot+'''" width="1200" height="800"></div>'''

    HTML+=addDesc('Sleep Routine during '+scatterplots['yearly'][-1].split('/')[0]+'-'+scatterplots['yearly'][0].split('/')[0]) +'''
    <div class="columnpadding"></div>
    <div class="imgcolumn">'''
    for plot in scatterplots['yearly']:
        HTML+='''<div class="gallery">
                <img src="'''+getb64Image(FilePath+plot)+'''"  alt="Sleep Time in '''+plot+'''" width="1200" height="800"></div>'''


    ### Monthly Plots : 
    #HTML+='</div>'+addDesc('Monthly Reports')+'<div class="imgcolumn">'
    HTML+=addDesc('Monthly Reports')
    monthPlots={}
    for pltYr in barplots['monthly']: # Create dicionary to reverse month plots for each year
       monthPlots.setdefault(pltYr.split('/')[0],[]) ; monthPlots[pltYr.split('/')[0]].append(pltYr)
    # Sleep Time
    for yr in monthPlots:
        HTML+=addDesc('Sleep Time in '+yr)
        for plot in monthPlots[yr][::-1]:
            HTML+='''<div class="gallery">
                 <img src="'''+getb64Image(FilePath+plot)+'''"  alt="Sleep Time in '''+plot+'''" width="1200" height="800"></div>'''
    HTML+='</div></div></div>' # End First Row & Column

    # Sleep Routine
    monthPlots={}
    for pltYr in scatterplots['monthly']: # Create dicionary to reverse month plots for each year
       monthPlots.setdefault(pltYr.split('/')[0],[]) ; monthPlots[pltYr.split('/')[0]].append(pltYr)
    for yr in monthPlots:
        HTML+=addDesc('Sleep Routine in '+yr)+'<div class="row"><div class="imgcol2">' # New Row for Each Year
        for plot in monthPlots[yr][::-1][::2]:
            HTML+='''<div class="gallery">
                 <img src="'''+getb64Image(FilePath+plot)+'''"  alt="Sleep Routine in '''+plot+'''" width="1200" height="800"></div>''' 
        HTML+='</div> <div class="imgcol2">' # End First Column & Add Second Column
        
        for plot in monthPlots[yr][::-1][1::2]:
            HTML+='''<div class="gallery">
                 <img src="'''+getb64Image(FilePath+plot)+'''"  alt="Sleep Routine in '''+plot+'''" width="1200" height="800"></div>'''         
        HTML+='</div></div>'

    sleepFile=ReportName+'Sleep Report  '+Period+'.html'
    with open(sleepFile,'w') as file:
        file.write(HTML)
    #Browser=r"EXCEL.EXE - Shortcut.lnk"+" "+filename
    print("\n   -- Opening '"+sleepFile+'" ..')
    os.system('start CHROME.EXE "'+sleepFile+'"')
    return sleepFile




######### Main Program ########
def android(file='MyActivity.json', apps=12, timezone='in', excludeapps=['com.miui.home' ], idealsleeptime=8, inlineimg=1,verbose=0):
    '''
    Generates Report of Sleep-Time, Sleep-Routine and App-Usage
    
    Steps to Get Android Activity File :
        1. Go to : https://takeout.google.com/?pli=1
        2. Under  "Select data to include", Click on "Deselect all"
        3. Scroll Down and Select "My Activity" . Click on "Multiple formats".   In  "Activity records, Choose 'JSON'  & then 'ok'.
        4. Scroll Down. Click on "Next Step". and then on "Create Export".
        5. Wait for the Google Data Download mail to arrive in your Gmail. Download the Zip file.

    Parameters
    ----------
    file : str, optional
        Pass MyActivity JSON file or Takeout zip file. The default is 'MyActivity.json'.
    apps : int or list , optional
        No. of Top Apps or List of Apps. The default is 12.
    timezone : str
        Pass the timezone region code. The default is 'in' for Indian Timezone.
    excludeapps : List
        List having app names to Exclude from App Usage. The default is ['com.miui.home' ].
    idealsleeptime : int
        Ideal Sleep Time. The default is 8.
    inlineimg : 0 or 1, 
        To include image in the Report iteself or not. The default is 1.
    verbose : 0 or 1, optional
        Shows Additional Progess during Report Generation. The default is 0.

    Returns
    -------
    Dictionary with Following Keys having Values 

    '''
    global TopAppsN, IdlSleepT, Timezone, InlineImg, Verbose
    TopAppsN=apps # Top Apps to Get App Usage for
    
    IdlSleepT=idealsleeptime
    Timezone=timezone
    InlineImg=inlineimg # Set Global variable
    Verbose=verbose
    ExcludeApps=excludeapps
    

    gData = creategDataDF(file) # Read json file into gData dictionary
    AppUsage, AppDailyUsage = appDataFrame(gData) # Create Two DataFrames from gData


    # Generate Sleep Report
    sleepdata = SleepData(AppDailyUsage) # Generates Sleep Data from DataFrame AppDailyUsage
    SleepYearlyMeanDF, SleepMonthlyMeanDF, barplots, scatterplots = SleepAnalysis(sleepdata) # Plots Monthwise and Gives Plot names
    sleepReport = generateSleepReportHTML(SleepYearlyMeanDF, SleepMonthlyMeanDF, barplots, scatterplots) # Generates Sleep Report HTML File

    # Generate App Usage Report
    AppMonthlyUsage = appUsageData(TopAppsN, AppUsage) # Using App
    AppAllTimeUsage, Plots = AppUsageAnalysis(AppMonthlyUsage,AppUsage) # Plot Yearly App Usage monthwise and Also Generates DataFrame having Yearly Stats
    AppUsageReport = generateAppUsageHTML(AppAllTimeUsage, Plots)

    dataDict={ 'AppUsage': AppUsage, 'AppDailyUsage': AppDailyUsage, 'SleepData': sleepdata,
               'SleepYearlyTable': SleepYearlyMeanDF, 'SleepMonthlyTable': SleepMonthlyMeanDF,
               'AppYearlyTable': AppAllTimeUsage, 'AppMonthlyTable': AppMonthlyUsage           }
    return  dataDict




# Global Variables
InlineImg=1
TopAppsN=12
Verbose=0
Timezone='in'

PlotStyle=''
ExcludeApps=['com.miui.home']


IdlSleepT = 8
timed={'sleep':[4,30], 'wake':[6,0]}

#analysegoogleactivity
if __name__ == '__main__' :
    android(apps=2)
    
    

