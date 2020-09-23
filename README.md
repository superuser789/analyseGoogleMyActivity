# analyseGoogleMyActivity
Generates Reports of Sleep Time, Sleep Routine and App Usage using Data from Google MyActivity : myactivity.google.com

Sleep Data and App Usage Data are generated on the basis of Following Assumptions :
  1. Sleep Data :
  
            Bed Time : Time at which User stops using phone and goes to bed.
            Wake Up Time : After bed time, the First Time at which User starts using phone.
            
  2. Time Spent on a App : Difference of Time between the opening time of the app and the next app.


### First Download Google Activity Data File by following these steps :

    1. Login to Google account. Go to : https://takeout.google.com/
    
    2. Under  "Select data to include", Click on "Deselect all" .
    
    3. Scroll Down and Select "My Activity" . Click on "Multiple formats".  In  "Activity records, Choose 'JSON'  & then 'ok'.
    
    4. Scroll Down. Click on "Next Step"  and then on "Create Export".
    
    5. Wait for the Google Data Download mail to arrive in your Gmail. Download the Zip file.
    
    
## Installation :
    pip install analysegooglemyactivity

 Requirements : "numpy", "pandas", "matplotlib"

## Usage :
   By Default, Looks for the Latest Takeout Zip in the current working directory
    
    from analyseGoogleMyActivity import androidReport
    reports = androidReport()

  Directly Pass the Takeout Zip to the parameter file ( Pass its Path also if the zip file is not in the current working directory)
   
    reports = androidReport(file='takeout-2020XXXXTXXXXXXZ-001.zip')

### Parameters : 
    file : str, optional
        Pass MyActivity JSON file or Takeout zip file with path. The default is 'MyActivity.json'.
    apps : int or list , optional
        No. of Top Apps or List of Apps to find usage for. The default is 12.
    timezone : str
        Pass the timezone region code. The default is 'in' for Indian Standard Time (IST).
    excludeapps : List
        List of app names to Exclude from App Usage calculation. The default is ['com.miui.home' ].
    idealsleeptime : int
        Ideal Sleep Time. The default is 8.
    inlineimg : 0 or 1, 
        To include image in the Report itself or not. The default is 1.
    showmarkerday: 0 or 1,
        To show day on each marker in sleep routine graphs. The default is 0.
    output : 0 or 1, optional
        If 1 , Returns Dictionary with Results in Pandas DataFrames, otherwise returns Reports names. The default is 1.
    yeartabs : 0 or 1, optional
        To Show Year & its Data in Tabs, The default is 1.
    verbose : 0 or 1, optional
        Shows Additional Progess during Report Generation. The default is 0.

    Returns
    Dictionary if Parameter output = 1   OR
    Tuple having Generated Report names if output = 0
    -------
    Dictionary with Following Keys having Values 
    'AppUsage': Time at which an App is opened, Pandas DataFrame
    'AppDailyUsage': Day wise data of App opened, Pandas DataFrame
    'SleepData': Bed Time & WakeUp Time with Sleep Time , Pandas DataFrame
    'SleepYearlyTable': Yearly Stats of Sleep Time & Sleep Routine
    'SleepMonthlyTable': Monthly Stats of Sleep Time & Sleep Routine
    'AppYearlyTable': Yearly App Usage Stats 
    'AppMonthlyTable': Monthly App Usage Stats
     __________________________________________________________
        
