#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.4),
    on Mon Nov 16 13:05:08 2020
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
prefs.hardware['audioLib'] = 'sounddevice'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.4'
expName = 'dummy_task'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sort_keys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/janaki/Dropbox/GeffenLab/Janaki/uncertainty code/dummy_task/dummy_task.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1280, 800], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "headphone_check_reminder"
headphone_check_reminderClock = core.Clock()
headphone_check_reminder_text = visual.TextStim(win=win, name='headphone_check_reminder_text',
    text='Please kindly wear headphones for our task, not wearing headphones will end up disqualifying you. ',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "volume_check_test"
volume_check_testClock = core.Clock()
volume_check_sound = sound.Sound('A', secs=-1, stereo=True, hamming=False,
    name='volume_check_sound')
volume_check_sound.setVolume(1.0)
volume_check_resp = keyboard.Keyboard()
volume_check_text = visual.TextStim(win=win, name='volume_check_text',
    text='Level calibration\n\nFirst, set your computer volume to about 25% of maximum.\n\nThen turn up the volume until the calibration noise is at a loud but comfortable level.\n\nPlease press ‘d’ once you are satisfied with the volume.',
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "headphone_check_test"
headphone_check_testClock = core.Clock()
headphone_check_sound = sound.Sound('A', secs=-1, stereo=True, hamming=True,
    name='headphone_check_sound')
headphone_check_sound.setVolume(1)
headphone_check_text = visual.TextStim(win=win, name='headphone_check_text',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
headphone_check_resp = keyboard.Keyboard()

# Initialize components for Routine "headphone_check_feedback"
headphone_check_feedbackClock = core.Clock()
headphone_count=0
expt=0
text_3 = visual.TextStim(win=win, name='text_3',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "intruction"
intructionClock = core.Clock()
Instructions = visual.TextStim(win=win, name='Instructions',
    text='Training: Please choose high (press key ‘h’) or low (press key ‘l’) based on what underlying distribution you believe the tone cloud (sequence of 3 tones) is from. \n\nAlso note that in the tone cloud any number of tones (0-3) could be from a background distribution. A background tone in most cases will sound different than one from the high/low distribution.\n\nWhen you encounter background tone(s) use your best judgement to decide if that entire set is from the low / high distribution. ‘h’ and ‘l’ are the only two options. \n\nPress any key to start.',
    font='Arial',
    pos=(0, 0), height=0.025, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp = keyboard.Keyboard()

# Initialize components for Routine "trial_train"
trial_trainClock = core.Clock()
tone_cloud = sound.Sound('A', secs=-1, stereo=True, hamming=True,
    name='tone_cloud')
tone_cloud.setVolume(0.3)
resp = keyboard.Keyboard()
reminder_instruction = visual.TextStim(win=win, name='reminder_instruction',
    text='default text',
    font='Arial',
    pos=(0, -1), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "feedback"
feedbackClock = core.Clock()
c = 0
w = 0
run = 0

feedback_text = visual.TextStim(win=win, name='feedback_text',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.08, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "instruction_test"
instruction_testClock = core.Clock()
test_text = visual.TextStim(win=win, name='test_text',
    text='Hopefully you have a clear idea of what a high or low tone sounds like and how it might be different from a background tone. \n\nBased on your knowledge, please start the testing phase. Press any key to begin.',
    font='Arial',
    pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
ready = keyboard.Keyboard()

# Initialize components for Routine "trial_test"
trial_testClock = core.Clock()
test_tone_cloud = sound.Sound('A', secs=-1, stereo=True, hamming=True,
    name='test_tone_cloud')
test_tone_cloud.setVolume(1.0)
test_resp = keyboard.Keyboard()
test_reminder_instruction = visual.TextStim(win=win, name='test_reminder_instruction',
    text='default text',
    font='Arial',
    pos=(0, -10), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "test_feedback"
test_feedbackClock = core.Clock()
""
c = 0
w = 0
run = 0
msg = ""
test_feedback_text = visual.TextStim(win=win, name='test_feedback_text',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "thanks"
thanksClock = core.Clock()
thanks_text = visual.TextStim(win=win, name='thanks_text',
    text='Thank you for your time and for helping us out!',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "headphone_check_reminder"-------
continueRoutine = True
routineTimer.add(4.000000)
# update component parameters for each repeat
# keep track of which components have finished
headphone_check_reminderComponents = [headphone_check_reminder_text]
for thisComponent in headphone_check_reminderComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
headphone_check_reminderClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "headphone_check_reminder"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = headphone_check_reminderClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=headphone_check_reminderClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *headphone_check_reminder_text* updates
    if headphone_check_reminder_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        headphone_check_reminder_text.frameNStart = frameN  # exact frame index
        headphone_check_reminder_text.tStart = t  # local t and not account for scr refresh
        headphone_check_reminder_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(headphone_check_reminder_text, 'tStartRefresh')  # time at next scr refresh
        headphone_check_reminder_text.setAutoDraw(True)
    if headphone_check_reminder_text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > headphone_check_reminder_text.tStartRefresh + 4-frameTolerance:
            # keep track of stop time/frame for later
            headphone_check_reminder_text.tStop = t  # not accounting for scr refresh
            headphone_check_reminder_text.frameNStop = frameN  # exact frame index
            win.timeOnFlip(headphone_check_reminder_text, 'tStopRefresh')  # time at next scr refresh
            headphone_check_reminder_text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in headphone_check_reminderComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "headphone_check_reminder"-------
for thisComponent in headphone_check_reminderComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('headphone_check_reminder_text.started', headphone_check_reminder_text.tStartRefresh)
thisExp.addData('headphone_check_reminder_text.stopped', headphone_check_reminder_text.tStopRefresh)

# set up handler to look after randomisation of conditions etc
volume_check_trials = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('Volume_check.csv'),
    seed=None, name='volume_check_trials')
thisExp.addLoop(volume_check_trials)  # add the loop to the experiment
thisVolume_check_trial = volume_check_trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisVolume_check_trial.rgb)
if thisVolume_check_trial != None:
    for paramName in thisVolume_check_trial:
        exec('{} = thisVolume_check_trial[paramName]'.format(paramName))

for thisVolume_check_trial in volume_check_trials:
    currentLoop = volume_check_trials
    # abbreviate parameter names if possible (e.g. rgb = thisVolume_check_trial.rgb)
    if thisVolume_check_trial != None:
        for paramName in thisVolume_check_trial:
            exec('{} = thisVolume_check_trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "volume_check_test"-------
    continueRoutine = True
    # update component parameters for each repeat
    volume_check_sound.setSound("assets/" + Name, hamming=False)
    volume_check_sound.setVolume(1, log=False)
    volume_check_resp.keys = []
    volume_check_resp.rt = []
    _volume_check_resp_allKeys = []
    # keep track of which components have finished
    volume_check_testComponents = [volume_check_sound, volume_check_resp, volume_check_text]
    for thisComponent in volume_check_testComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    volume_check_testClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "volume_check_test"-------
    while continueRoutine:
        # get current time
        t = volume_check_testClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=volume_check_testClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # start/stop volume_check_sound
        if volume_check_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            volume_check_sound.frameNStart = frameN  # exact frame index
            volume_check_sound.tStart = t  # local t and not account for scr refresh
            volume_check_sound.tStartRefresh = tThisFlipGlobal  # on global time
            volume_check_sound.play(when=win)  # sync with win flip
        
        # *volume_check_resp* updates
        waitOnFlip = False
        if volume_check_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            volume_check_resp.frameNStart = frameN  # exact frame index
            volume_check_resp.tStart = t  # local t and not account for scr refresh
            volume_check_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(volume_check_resp, 'tStartRefresh')  # time at next scr refresh
            volume_check_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(volume_check_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(volume_check_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if volume_check_resp.status == STARTED and not waitOnFlip:
            theseKeys = volume_check_resp.getKeys(keyList=['d'], waitRelease=False)
            _volume_check_resp_allKeys.extend(theseKeys)
            if len(_volume_check_resp_allKeys):
                volume_check_resp.keys = _volume_check_resp_allKeys[-1].name  # just the last key pressed
                volume_check_resp.rt = _volume_check_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *volume_check_text* updates
        if volume_check_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            volume_check_text.frameNStart = frameN  # exact frame index
            volume_check_text.tStart = t  # local t and not account for scr refresh
            volume_check_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(volume_check_text, 'tStartRefresh')  # time at next scr refresh
            volume_check_text.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in volume_check_testComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "volume_check_test"-------
    for thisComponent in volume_check_testComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    volume_check_sound.stop()  # ensure sound has stopped at end of routine
    volume_check_trials.addData('volume_check_sound.started', volume_check_sound.tStartRefresh)
    volume_check_trials.addData('volume_check_sound.stopped', volume_check_sound.tStopRefresh)
    volume_check_trials.addData('volume_check_text.started', volume_check_text.tStartRefresh)
    volume_check_trials.addData('volume_check_text.stopped', volume_check_text.tStopRefresh)
    # the Routine "volume_check_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1 repeats of 'volume_check_trials'

# get names of stimulus parameters
if volume_check_trials.trialList in ([], [None], None):
    params = []
else:
    params = volume_check_trials.trialList[0].keys()
# save data for this loop
volume_check_trials.saveAsExcel(filename + '.xlsx', sheetName='volume_check_trials',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])
volume_check_trials.saveAsText(filename + 'volume_check_trials.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# set up handler to look after randomisation of conditions etc
headphone_check_trials = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('headphone_check.csv'),
    seed=None, name='headphone_check_trials')
thisExp.addLoop(headphone_check_trials)  # add the loop to the experiment
thisHeadphone_check_trial = headphone_check_trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisHeadphone_check_trial.rgb)
if thisHeadphone_check_trial != None:
    for paramName in thisHeadphone_check_trial:
        exec('{} = thisHeadphone_check_trial[paramName]'.format(paramName))

for thisHeadphone_check_trial in headphone_check_trials:
    currentLoop = headphone_check_trials
    # abbreviate parameter names if possible (e.g. rgb = thisHeadphone_check_trial.rgb)
    if thisHeadphone_check_trial != None:
        for paramName in thisHeadphone_check_trial:
            exec('{} = thisHeadphone_check_trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "headphone_check_test"-------
    continueRoutine = True
    # update component parameters for each repeat
    headphone_check_sound.setSound("assets/" + Name, hamming=True)
    headphone_check_sound.setVolume(1, log=False)
    headphone_check_text.setText('Please enter 1 / 2 / 3 based on which tone you think is the softest. ')
    headphone_check_resp.keys = []
    headphone_check_resp.rt = []
    _headphone_check_resp_allKeys = []
    # keep track of which components have finished
    headphone_check_testComponents = [headphone_check_sound, headphone_check_text, headphone_check_resp]
    for thisComponent in headphone_check_testComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    headphone_check_testClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "headphone_check_test"-------
    while continueRoutine:
        # get current time
        t = headphone_check_testClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=headphone_check_testClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # start/stop headphone_check_sound
        if headphone_check_sound.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            headphone_check_sound.frameNStart = frameN  # exact frame index
            headphone_check_sound.tStart = t  # local t and not account for scr refresh
            headphone_check_sound.tStartRefresh = tThisFlipGlobal  # on global time
            headphone_check_sound.play(when=win)  # sync with win flip
        
        # *headphone_check_text* updates
        if headphone_check_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            headphone_check_text.frameNStart = frameN  # exact frame index
            headphone_check_text.tStart = t  # local t and not account for scr refresh
            headphone_check_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(headphone_check_text, 'tStartRefresh')  # time at next scr refresh
            headphone_check_text.setAutoDraw(True)
        
        # *headphone_check_resp* updates
        waitOnFlip = False
        if headphone_check_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            headphone_check_resp.frameNStart = frameN  # exact frame index
            headphone_check_resp.tStart = t  # local t and not account for scr refresh
            headphone_check_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(headphone_check_resp, 'tStartRefresh')  # time at next scr refresh
            headphone_check_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(headphone_check_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(headphone_check_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if headphone_check_resp.status == STARTED and not waitOnFlip:
            theseKeys = headphone_check_resp.getKeys(keyList=['1', '2', '3'], waitRelease=False)
            _headphone_check_resp_allKeys.extend(theseKeys)
            if len(_headphone_check_resp_allKeys):
                headphone_check_resp.keys = _headphone_check_resp_allKeys[-1].name  # just the last key pressed
                headphone_check_resp.rt = _headphone_check_resp_allKeys[-1].rt
                # was this correct?
                if (headphone_check_resp.keys == str(corrAns)) or (headphone_check_resp.keys == corrAns):
                    headphone_check_resp.corr = 1
                else:
                    headphone_check_resp.corr = 0
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in headphone_check_testComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "headphone_check_test"-------
    for thisComponent in headphone_check_testComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    headphone_check_sound.stop()  # ensure sound has stopped at end of routine
    headphone_check_trials.addData('headphone_check_sound.started', headphone_check_sound.tStartRefresh)
    headphone_check_trials.addData('headphone_check_sound.stopped', headphone_check_sound.tStopRefresh)
    headphone_check_trials.addData('headphone_check_text.started', headphone_check_text.tStartRefresh)
    headphone_check_trials.addData('headphone_check_text.stopped', headphone_check_text.tStopRefresh)
    # check responses
    if headphone_check_resp.keys in ['', [], None]:  # No response was made
        headphone_check_resp.keys = None
        # was no response the correct answer?!
        if str(corrAns).lower() == 'none':
           headphone_check_resp.corr = 1;  # correct non-response
        else:
           headphone_check_resp.corr = 0;  # failed to respond (incorrectly)
    # store data for headphone_check_trials (TrialHandler)
    headphone_check_trials.addData('headphone_check_resp.keys',headphone_check_resp.keys)
    headphone_check_trials.addData('headphone_check_resp.corr', headphone_check_resp.corr)
    if headphone_check_resp.keys != None:  # we had a response
        headphone_check_trials.addData('headphone_check_resp.rt', headphone_check_resp.rt)
    headphone_check_trials.addData('headphone_check_resp.started', headphone_check_resp.tStartRefresh)
    headphone_check_trials.addData('headphone_check_resp.stopped', headphone_check_resp.tStopRefresh)
    # the Routine "headphone_check_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "headphone_check_feedback"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    if not headphone_check_resp.keys :
      headphone_check_msg="Failed to respond"
    elif headphone_check_resp.corr:#stored on last run routine
      headphone_count +=1 
      headphone_check_msg = "Correct answers: " + str(headphone_count)
    expt += 1
    text_3.setText(headphone_count)
    # keep track of which components have finished
    headphone_check_feedbackComponents = [text_3]
    for thisComponent in headphone_check_feedbackComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    headphone_check_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "headphone_check_feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = headphone_check_feedbackClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=headphone_check_feedbackClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            text_3.setAutoDraw(True)
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_3, 'tStopRefresh')  # time at next scr refresh
                text_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in headphone_check_feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "headphone_check_feedback"-------
    for thisComponent in headphone_check_feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    if expt == 6 and headphone_count < 5:
        psychoJS.quit()
    headphone_check_trials.addData('text_3.started', text_3.tStartRefresh)
    headphone_check_trials.addData('text_3.stopped', text_3.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1 repeats of 'headphone_check_trials'

# get names of stimulus parameters
if headphone_check_trials.trialList in ([], [None], None):
    params = []
else:
    params = headphone_check_trials.trialList[0].keys()
# save data for this loop
headphone_check_trials.saveAsExcel(filename + '.xlsx', sheetName='headphone_check_trials',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])
headphone_check_trials.saveAsText(filename + 'headphone_check_trials.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "intruction"-------
continueRoutine = True
# update component parameters for each repeat
key_resp.keys = []
key_resp.rt = []
_key_resp_allKeys = []
# keep track of which components have finished
intructionComponents = [Instructions, key_resp]
for thisComponent in intructionComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
intructionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "intruction"-------
while continueRoutine:
    # get current time
    t = intructionClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=intructionClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Instructions* updates
    if Instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Instructions.frameNStart = frameN  # exact frame index
        Instructions.tStart = t  # local t and not account for scr refresh
        Instructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Instructions, 'tStartRefresh')  # time at next scr refresh
        Instructions.setAutoDraw(True)
    
    # *key_resp* updates
    waitOnFlip = False
    if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp.frameNStart = frameN  # exact frame index
        key_resp.tStart = t  # local t and not account for scr refresh
        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
        key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp.status == STARTED and not waitOnFlip:
        theseKeys = key_resp.getKeys(keyList=None, waitRelease=False)
        _key_resp_allKeys.extend(theseKeys)
        if len(_key_resp_allKeys):
            key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
            key_resp.rt = _key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in intructionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "intruction"-------
for thisComponent in intructionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Instructions.started', Instructions.tStartRefresh)
thisExp.addData('Instructions.stopped', Instructions.tStopRefresh)
# the Routine "intruction" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('train_three_tones_v2_file.csv'),
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial_train"-------
    continueRoutine = True
    # update component parameters for each repeat
    tone_cloud.setSound("soundfiles_three_tones_train_v2/"+ Name, hamming=True)
    tone_cloud.setVolume(0.3, log=False)
    resp.keys = []
    resp.rt = []
    _resp_allKeys = []
    reminder_instruction.setText('keys - ‘h’ : high dist     ’l’ : low dist')
    # keep track of which components have finished
    trial_trainComponents = [tone_cloud, resp, reminder_instruction]
    for thisComponent in trial_trainComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trial_trainClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial_train"-------
    while continueRoutine:
        # get current time
        t = trial_trainClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trial_trainClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # start/stop tone_cloud
        if tone_cloud.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            tone_cloud.frameNStart = frameN  # exact frame index
            tone_cloud.tStart = t  # local t and not account for scr refresh
            tone_cloud.tStartRefresh = tThisFlipGlobal  # on global time
            tone_cloud.play(when=win)  # sync with win flip
        
        # *resp* updates
        waitOnFlip = False
        if resp.status == NOT_STARTED and tThisFlip >= 0.6-frameTolerance:
            # keep track of start time/frame for later
            resp.frameNStart = frameN  # exact frame index
            resp.tStart = t  # local t and not account for scr refresh
            resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
            resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if resp.status == STARTED and not waitOnFlip:
            theseKeys = resp.getKeys(keyList=['h', 'l'], waitRelease=False)
            _resp_allKeys.extend(theseKeys)
            if len(_resp_allKeys):
                resp.keys = _resp_allKeys[-1].name  # just the last key pressed
                resp.rt = _resp_allKeys[-1].rt
                # was this correct?
                if (resp.keys == str(corrAns)) or (resp.keys == corrAns):
                    resp.corr = 1
                else:
                    resp.corr = 0
                # a response ends the routine
                continueRoutine = False
        
        # *reminder_instruction* updates
        if reminder_instruction.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            reminder_instruction.frameNStart = frameN  # exact frame index
            reminder_instruction.tStart = t  # local t and not account for scr refresh
            reminder_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reminder_instruction, 'tStartRefresh')  # time at next scr refresh
            reminder_instruction.setAutoDraw(True)
        if reminder_instruction.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reminder_instruction.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                reminder_instruction.tStop = t  # not accounting for scr refresh
                reminder_instruction.frameNStop = frameN  # exact frame index
                win.timeOnFlip(reminder_instruction, 'tStopRefresh')  # time at next scr refresh
                reminder_instruction.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_trainComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial_train"-------
    for thisComponent in trial_trainComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    tone_cloud.stop()  # ensure sound has stopped at end of routine
    trials.addData('tone_cloud.started', tone_cloud.tStartRefresh)
    trials.addData('tone_cloud.stopped', tone_cloud.tStopRefresh)
    # check responses
    if resp.keys in ['', [], None]:  # No response was made
        resp.keys = None
        # was no response the correct answer?!
        if str(corrAns).lower() == 'none':
           resp.corr = 1;  # correct non-response
        else:
           resp.corr = 0;  # failed to respond (incorrectly)
    # store data for trials (TrialHandler)
    trials.addData('resp.keys',resp.keys)
    trials.addData('resp.corr', resp.corr)
    if resp.keys != None:  # we had a response
        trials.addData('resp.rt', resp.rt)
    trials.addData('resp.started', resp.tStartRefresh)
    trials.addData('resp.stopped', resp.tStopRefresh)
    trials.addData('reminder_instruction.started', reminder_instruction.tStartRefresh)
    trials.addData('reminder_instruction.stopped', reminder_instruction.tStopRefresh)
    # the Routine "trial_train" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "feedback"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    if not resp.keys :
      msg="Failed to respond"
    elif resp.corr:#stored on last run routine
      msg="Correct!" 
      c = c+1
    else:
      msg="Not quite!"
      w = w+1
      
    run = run+1 
    if run %10 == 0:
       msg = msg + "       " + "last 10 trials   " + str(c*100/(c+w)) + " % correct"
       c = 0
       w = 0
       
    
    feedback_text.setText(msg)
    # keep track of which components have finished
    feedbackComponents = [feedback_text]
    for thisComponent in feedbackComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = feedbackClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=feedbackClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *feedback_text* updates
        if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            feedback_text.frameNStart = frameN  # exact frame index
            feedback_text.tStart = t  # local t and not account for scr refresh
            feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
            feedback_text.setAutoDraw(True)
        if feedback_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > feedback_text.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                feedback_text.tStop = t  # not accounting for scr refresh
                feedback_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(feedback_text, 'tStopRefresh')  # time at next scr refresh
                feedback_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "feedback"-------
    for thisComponent in feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('feedback_text.started', feedback_text.tStartRefresh)
    trials.addData('feedback_text.stopped', feedback_text.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1 repeats of 'trials'

# get names of stimulus parameters
if trials.trialList in ([], [None], None):
    params = []
else:
    params = trials.trialList[0].keys()
# save data for this loop
trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])
trials.saveAsText(filename + 'trials.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "instruction_test"-------
continueRoutine = True
# update component parameters for each repeat
ready.keys = []
ready.rt = []
_ready_allKeys = []
# keep track of which components have finished
instruction_testComponents = [test_text, ready]
for thisComponent in instruction_testComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_testClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_test"-------
while continueRoutine:
    # get current time
    t = instruction_testClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_testClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *test_text* updates
    if test_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        test_text.frameNStart = frameN  # exact frame index
        test_text.tStart = t  # local t and not account for scr refresh
        test_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(test_text, 'tStartRefresh')  # time at next scr refresh
        test_text.setAutoDraw(True)
    
    # *ready* updates
    waitOnFlip = False
    if ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        ready.frameNStart = frameN  # exact frame index
        ready.tStart = t  # local t and not account for scr refresh
        ready.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(ready, 'tStartRefresh')  # time at next scr refresh
        ready.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(ready.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(ready.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if ready.status == STARTED and not waitOnFlip:
        theseKeys = ready.getKeys(keyList=None, waitRelease=False)
        _ready_allKeys.extend(theseKeys)
        if len(_ready_allKeys):
            ready.keys = _ready_allKeys[-1].name  # just the last key pressed
            ready.rt = _ready_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_testComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_test"-------
for thisComponent in instruction_testComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('test_text.started', test_text.tStartRefresh)
thisExp.addData('test_text.stopped', test_text.tStopRefresh)
# the Routine "instruction_test" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
test_trials = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('test_three_tones_v2_file.csv'),
    seed=None, name='test_trials')
thisExp.addLoop(test_trials)  # add the loop to the experiment
thisTest_trial = test_trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTest_trial.rgb)
if thisTest_trial != None:
    for paramName in thisTest_trial:
        exec('{} = thisTest_trial[paramName]'.format(paramName))

for thisTest_trial in test_trials:
    currentLoop = test_trials
    # abbreviate parameter names if possible (e.g. rgb = thisTest_trial.rgb)
    if thisTest_trial != None:
        for paramName in thisTest_trial:
            exec('{} = thisTest_trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial_test"-------
    continueRoutine = True
    # update component parameters for each repeat
    test_tone_cloud.setSound("soundfiles_three_tones_test_v2/"+ Name, hamming=True)
    test_tone_cloud.setVolume(0.3, log=False)
    test_resp.keys = []
    test_resp.rt = []
    _test_resp_allKeys = []
    # keep track of which components have finished
    trial_testComponents = [test_tone_cloud, test_resp, test_reminder_instruction]
    for thisComponent in trial_testComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trial_testClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial_test"-------
    while continueRoutine:
        # get current time
        t = trial_testClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trial_testClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # start/stop test_tone_cloud
        if test_tone_cloud.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            test_tone_cloud.frameNStart = frameN  # exact frame index
            test_tone_cloud.tStart = t  # local t and not account for scr refresh
            test_tone_cloud.tStartRefresh = tThisFlipGlobal  # on global time
            test_tone_cloud.play(when=win)  # sync with win flip
        
        # *test_resp* updates
        waitOnFlip = False
        if test_resp.status == NOT_STARTED and tThisFlip >= 0.6-frameTolerance:
            # keep track of start time/frame for later
            test_resp.frameNStart = frameN  # exact frame index
            test_resp.tStart = t  # local t and not account for scr refresh
            test_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(test_resp, 'tStartRefresh')  # time at next scr refresh
            test_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(test_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(test_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if test_resp.status == STARTED and not waitOnFlip:
            theseKeys = test_resp.getKeys(keyList=['h', 'l'], waitRelease=False)
            _test_resp_allKeys.extend(theseKeys)
            if len(_test_resp_allKeys):
                test_resp.keys = _test_resp_allKeys[-1].name  # just the last key pressed
                test_resp.rt = _test_resp_allKeys[-1].rt
                # was this correct?
                if (test_resp.keys == str(corrAns)) or (test_resp.keys == corrAns):
                    test_resp.corr = 1
                else:
                    test_resp.corr = 0
                # a response ends the routine
                continueRoutine = False
        
        # *test_reminder_instruction* updates
        if test_reminder_instruction.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            test_reminder_instruction.frameNStart = frameN  # exact frame index
            test_reminder_instruction.tStart = t  # local t and not account for scr refresh
            test_reminder_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(test_reminder_instruction, 'tStartRefresh')  # time at next scr refresh
            test_reminder_instruction.setAutoDraw(True)
        if test_reminder_instruction.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > test_reminder_instruction.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                test_reminder_instruction.tStop = t  # not accounting for scr refresh
                test_reminder_instruction.frameNStop = frameN  # exact frame index
                win.timeOnFlip(test_reminder_instruction, 'tStopRefresh')  # time at next scr refresh
                test_reminder_instruction.setAutoDraw(False)
        if test_reminder_instruction.status == STARTED:  # only update if drawing
            test_reminder_instruction.setText('‘h’: high dist,     ‘l’: low dist', log=False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_testComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial_test"-------
    for thisComponent in trial_testComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    test_tone_cloud.stop()  # ensure sound has stopped at end of routine
    test_trials.addData('test_tone_cloud.started', test_tone_cloud.tStartRefresh)
    test_trials.addData('test_tone_cloud.stopped', test_tone_cloud.tStopRefresh)
    # check responses
    if test_resp.keys in ['', [], None]:  # No response was made
        test_resp.keys = None
        # was no response the correct answer?!
        if str(corrAns).lower() == 'none':
           test_resp.corr = 1;  # correct non-response
        else:
           test_resp.corr = 0;  # failed to respond (incorrectly)
    # store data for test_trials (TrialHandler)
    test_trials.addData('test_resp.keys',test_resp.keys)
    test_trials.addData('test_resp.corr', test_resp.corr)
    if test_resp.keys != None:  # we had a response
        test_trials.addData('test_resp.rt', test_resp.rt)
    test_trials.addData('test_resp.started', test_resp.tStartRefresh)
    test_trials.addData('test_resp.stopped', test_resp.tStopRefresh)
    test_trials.addData('test_reminder_instruction.started', test_reminder_instruction.tStartRefresh)
    test_trials.addData('test_reminder_instruction.stopped', test_reminder_instruction.tStopRefresh)
    # the Routine "trial_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "test_feedback"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    if test_resp.corr:#stored on last run routine
      c = c+1
      msg_test="Correct!"
    else:
      w = w+1  
      msg_test="Oops! That was wrong"
     
    run = run+1 
    if run %10 == 0:
       msg =  "Last 10 trials   " + str(c*100/(c+w)) + " % correct"
       c = 0
       w = 0
    else:
        msg = ""
    test_feedback_text.setText(msg)
    # keep track of which components have finished
    test_feedbackComponents = [test_feedback_text]
    for thisComponent in test_feedbackComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    test_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "test_feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = test_feedbackClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=test_feedbackClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *test_feedback_text* updates
        if test_feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            test_feedback_text.frameNStart = frameN  # exact frame index
            test_feedback_text.tStart = t  # local t and not account for scr refresh
            test_feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(test_feedback_text, 'tStartRefresh')  # time at next scr refresh
            test_feedback_text.setAutoDraw(True)
        if test_feedback_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > test_feedback_text.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                test_feedback_text.tStop = t  # not accounting for scr refresh
                test_feedback_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(test_feedback_text, 'tStopRefresh')  # time at next scr refresh
                test_feedback_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in test_feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "test_feedback"-------
    for thisComponent in test_feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    test_trials.addData('test_feedback_text.started', test_feedback_text.tStartRefresh)
    test_trials.addData('test_feedback_text.stopped', test_feedback_text.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1 repeats of 'test_trials'

# get names of stimulus parameters
if test_trials.trialList in ([], [None], None):
    params = []
else:
    params = test_trials.trialList[0].keys()
# save data for this loop
test_trials.saveAsExcel(filename + '.xlsx', sheetName='test_trials',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])
test_trials.saveAsText(filename + 'test_trials.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "thanks"-------
continueRoutine = True
routineTimer.add(10.000000)
# update component parameters for each repeat
# keep track of which components have finished
thanksComponents = [thanks_text]
for thisComponent in thanksComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
thanksClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "thanks"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = thanksClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=thanksClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *thanks_text* updates
    if thanks_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        thanks_text.frameNStart = frameN  # exact frame index
        thanks_text.tStart = t  # local t and not account for scr refresh
        thanks_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(thanks_text, 'tStartRefresh')  # time at next scr refresh
        thanks_text.setAutoDraw(True)
    if thanks_text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > thanks_text.tStartRefresh + 10-frameTolerance:
            # keep track of stop time/frame for later
            thanks_text.tStop = t  # not accounting for scr refresh
            thanks_text.frameNStop = frameN  # exact frame index
            win.timeOnFlip(thanks_text, 'tStopRefresh')  # time at next scr refresh
            thanks_text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in thanksComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "thanks"-------
for thisComponent in thanksComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('thanks_text.started', thanks_text.tStartRefresh)
thisExp.addData('thanks_text.stopped', thanks_text.tStopRefresh)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
