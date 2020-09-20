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

from analyseGoogleMyActivity.android import android