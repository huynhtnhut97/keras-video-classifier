#! /usr/bin/python
# -*- coding: utf-8 -*-

#
# tkinter example for VLC Python bindings
# Copyright (C) 2015 the VideoLAN team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
#
"""A simple example for VLC python bindings using tkinter. Uses python 3.4

Author: Patrick Fay
Date: 23-09-2015
"""

# import external libraries
import vlc
import sys

if sys.version_info[0] < 3:
	import Tkinter as Tk
	from Tkinter import ttk
	from Tkinter.filedialog import askopenfilename
else:
	import tkinter as Tk
	from tkinter import ttk
	from tkinter.filedialog import askopenfilename
	from tkinter import messagebox
	from tkinter import PhotoImage

# import standard libraries
import os
import pathlib
from threading import Thread, Event
import time
import platform
import numpy as np
import sys
import os
import cv2
import datetime
import time
import subprocess
from PIL import Image, ImageTk
class ttkTimer(Thread):
	"""a class serving same function as wxTimer... but there may be better ways to do this
	"""
	def __init__(self, callback, tick):
		Thread.__init__(self)
		self.callback = callback
		self.stopFlag = Event()
		self.tick = tick
		self.iters = 0

	def run(self):
		while not self.stopFlag.wait(self.tick):
			self.iters += 1
			self.callback()

	def stop(self):
		self.stopFlag.set()

	def get(self):
		return self.iters

class Player(Tk.Frame):
	"""The main window has to deal with events.
	"""
	def __init__(self, parent, title=None):
		Tk.Frame.__init__(self, parent)

		self.parent = parent

		if title == None:
			title = "tk_vlc"
		self.parent.title(title)

		# Menu Bar
		#   File Menu
		menubar = Tk.Menu(self.parent)
		self.parent.config(menu=menubar)

		folder_icon = ImageTk.PhotoImage(Image.open('./Icon/folder-icon.png'))
		play_icon = ImageTk.PhotoImage(Image.open('./Icon/control-play-icon.png'))
		stop_icon = ImageTk.PhotoImage(Image.open('./Icon/control-stop-icon.png'))
		pause_icon = ImageTk.PhotoImage(Image.open('./Icon/control-pause-icon.png'))
		volume_icon = ImageTk.PhotoImage(Image.open('./Icon/volume-icon.png'))



		fileMenu = Tk.Menu(menubar)
		fileMenu.add_command(label="Open",underline=0, command=self.OnOpen)
		fileMenu.add_command(label="Detect",underline=1, command=self.OnDetect)
		fileMenu.add_separator()
		fileMenu.add_command(label="Exit", underline=2, command=_quit)
		menubar.add_cascade(label="File", menu=fileMenu)

		# The second panel holds controls
		self.player = None
		self.videopanel = ttk.Frame(self.parent)
		self.canvas = Tk.Canvas(self.videopanel).pack(fill=Tk.BOTH,expand=1)
		self.videopanel.pack(fill=Tk.BOTH,expand=1)
		#self.controller = controller

		ctrlpanel = ttk.Frame(self.parent)
		pause  = ttk.Button(ctrlpanel, image=pause_icon, command=self.OnPause)
		play   = ttk.Button(ctrlpanel, image=play_icon, command=self.OnPlay)
		stop   = ttk.Button(ctrlpanel, image=stop_icon, command=self.OnStop)
		volume = ttk.Button(ctrlpanel, image=volume_icon, command=self.OnSetVolume)
		self.progress = ttk.Progressbar(ctrlpanel, orient="horizontal", length=200, mode="determinate")
		#self.progress.bind('<Map>',self.OnDetect)
		self.bytes = 0
		self.maxbytes = 0

		pause.image = pause_icon
		play.image = play_icon
		stop.image = stop_icon
		volume.image = volume_icon

		self.progress.pack(side=Tk.LEFT)
		pause.pack(side=Tk.LEFT)
		play.pack(side=Tk.LEFT)
		stop.pack(side=Tk.LEFT)
		volume.pack(side=Tk.LEFT)
		self.volume_var = Tk.IntVar()
		self.volslider = Tk.Scale(ctrlpanel, variable=self.volume_var, command=self.volume_sel,
				from_=0, to=100, orient=Tk.HORIZONTAL, length=100)
		self.volslider.pack(side=Tk.LEFT)
		ctrlpanel.pack(side=Tk.BOTTOM)

		ctrlpanel2 = ttk.Frame(self.parent)
		self.scale_var = Tk.DoubleVar()
		self.timeslider_last_val = ""
		self.timeslider = Tk.Scale(ctrlpanel2, variable=self.scale_var, command=self.scale_sel,
				from_=0, to=1000, orient=Tk.HORIZONTAL, length=500)
		self.timeslider.pack(side=Tk.BOTTOM, fill=Tk.X,expand=1)
		self.timeslider_last_update = time.time()
		ctrlpanel2.pack(side=Tk.BOTTOM,fill=Tk.X)


		# VLC player controls
		self.Instance = vlc.Instance()
		self.player = self.Instance.media_player_new()

		# below is a test, now use the File->Open file menu
		#media = self.Instance.media_new('output.mp4')
		#self.player.set_media(media)
		#self.player.play() # hit the player button
		#self.player.video_set_deinterlace(str_to_bytes('yadif'))

		self.timer = ttkTimer(self.OnTimer, 1.0)
		self.timer.start()
		self.parent.update()

		#self.player.set_hwnd(self.GetHandle()) # for windows, OnOpen does does this


	def OnExit(self, evt):
		"""Closes the window.
		"""
		self.Close()
	def Detect(self, fullname):
		sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

		from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
		from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels

		vgg16_include_top = True
		data_dir_path = os.path.join(os.path.dirname(__file__), 'very_large_data')
		model_dir_path = os.path.join(os.path.dirname(__file__), 'models', 'UCF-101')
		demo_dir_path = os.path.join(os.path.dirname(__file__), 'real-data')
		config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
																					  vgg16_include_top=vgg16_include_top)
		weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
																					  vgg16_include_top=vgg16_include_top)

		np.random.seed(42)

		load_ucf(data_dir_path)

		predictor = VGG16BidirectionalLSTMVideoClassifier()
		predictor.load_model(config_file_path, weight_file_path)

		print('reaching here three')
		
		unusual_actions = []
		usual_actions = []
		interval = 3
		txtFile = open('./reports/prediction_Each_Interval_Report'+'.txt','w')
		
		if os.path.isfile(fullname):
			print("Predicting video: {}".format(os.path.basename(fullname)))
			predicted_labels = predictor.predict(fullname,interval=interval)
			print("TOTAL {} ACTIONS".format(len(predicted_labels)))
			for index in range(len(predicted_labels)):
				if(predicted_labels[index] == "Unusual action"):
					time = str(datetime.timedelta(seconds=index))
					print('predicted: ' + predicted_labels[index] + ' at: {0}'.format(time))
					unusual_actions.append(list((fullname,time)))
				elif(predicted_labels[index] == "Usual action"):
					time = str(datetime.timedelta(seconds=index))
					print('predicted: ' + predicted_labels[index] + ' at: {0}'.format(time))
					usual_actions.append(list((fullname,time)))
		print("TOTAL: {} unusual actions and {} usual actions".format(len(unusual_actions),len(usual_actions)))
		txtFile.write("Unsual actions")
		txtFile.write('\n')
		for item in unusual_actions:
			txtFile.write(item[0] + ' at '+ item[1])
			txtFile.write('\n')
		txtFile.write("Usual actions")
		txtFile.write('\n')
		for item in usual_actions:
			txtFile.write(item[0] + ' at '+ item[1])
			txtFile.write('\n')
		txtFile.close()
		self.SplitVideo(fullname,unusual_actions)
		return self.DrawToVideo(fullname,unusual_actions)
	def SplitVideo(self,fullname, unusuals):
		index = 1
		self.progress['maximum'] = len(unusuals)

		for item in unusuals:
			times = time.strptime(item[1],'%H:%M:%S')
			second = int(datetime.timedelta(hours=times.tm_hour, minutes=times.tm_min, seconds=times.tm_sec).total_seconds())
			if not os.path.exists('./results'):
				os.mkdir('./results')
			filename = './results/unusual_action_{}_{}.avi'.format(os.path.splitext(os.path.basename(fullname))[0],index)
			start = second
			end = second + 2
			cmd = "ffmpeg -y -i {} -ss {} -t {} {}".format(fullname,start,3,filename)
			subprocess.run(cmd, stderr=subprocess.STDOUT)
			self.progress["value"]=index
			self.progress.update()
			index+=1
	def DrawToVideo(self,fullname, unusuals):


		cap = cv2.VideoCapture(fullname)
		# Find OpenCV version
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		if int(major_ver)  < 3 :
			fps = cap.get(cv2.CV_CAP_PROP_FPS)
			print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
		else:
			fps = cap.get(cv2.CAP_PROP_FPS)
			print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
		length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		self.progress['maximum'] = length
		print("Total frames: {}".format(length))
		# _, frame = cap.read()
		# height, width, _ = frame.shape
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		videoWriter = cv2.VideoWriter('./output_{}.avi'.format(os.path.basename(fullname)), fourcc, fps, (int(width), int(height)))
		assert cap.isOpened(), 'Cannot capture source'

		frames = 0
		
		# while cap.isOpened():
		# 	ret, frame = cap.read()
		# 	if ret:
		# 		for action in unusuals:
		# 			second = item[1]
		# 			if (frames<second*fps):
		# 				cv2.puttext(frame, "Unusual action detected",(10,10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1)
		# 				videoWriter.write(frame)
		# 			elif(frames==second*fps):
		# 				unusuals.remove(action)
		# 				break;
		# 	frames +=1
		for action in unusuals:
			times = time.strptime(action[1],'%H:%M:%S')
			second = int(datetime.timedelta(hours=times.tm_hour, minutes=times.tm_min, seconds=times.tm_sec).total_seconds())
			print("Drawing at: {}".format(action[1]))
			while cap.isOpened():
				ret, frame = cap.read()
				if ret:
					if ((frames-second*int(fps)<=int(fps)*3) and (frames>=second*int(fps))):
						cv2.putText(frame, "Unusual action detected",(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),5)
						videoWriter.write(frame)
					#elif((frames-second*int(fps)>int(fps)*3) and (frames>=second*int(fps))):
					elif((frames-second*int(fps)>int(fps)*3) and (frames>=second*int(fps))):
						break;
					else:
						videoWriter.write(frame)
						#unusuals.remove(action)
					self.progress["value"]=frames
					self.progress.update()
					frames +=1
				else:
					videoWriter.release()
					break;
				#print(frames)
				
		# while cap.isOpened():
		# 	ret, frame = cap.read()
		# 	if ret:
		# 		videoWriter.write(frame)
		# 	else:
		# 		videoWriter.release()
		output = './output_{}.avi'.format(os.path.basename(fullname))
		return output
	def OnDetect(self):
		"""Pop up a new dialow window to choose a file, then play the selected file.
		"""
		# if a file is already running, then stop it.
		self.OnStop()

		# Create a file dialog opened in the current home directory, where
		# you can display all kind of files, having as title "Choose a file".
		p = pathlib.Path(os.path.expanduser("~"))
		fullname =  askopenfilename(initialdir = p, title = "choose your file",filetypes = (("all files","*.*"),("mp4 files","*.mp4")))
		fullname = self.Detect(fullname)
		if os.path.isfile(fullname):
			dirname  = os.path.dirname(fullname)
			filename = os.path.basename(fullname)
			# Creation
			self.Media = self.Instance.media_new(str(os.path.join(dirname, filename)))
			self.player.set_media(self.Media)
			# Report the title of the file chosen
			#title = self.player.get_title()
			#  if an error was encountred while retriving the title, then use
			#  filename
			#if title == -1:
			#    title = filename
			#self.SetTitle("%s - tkVLCplayer" % title)

			# set the window id where to render VLC's video output
			if platform.system() == 'Windows':
				self.player.set_hwnd(self.GetHandle())
			else:
				self.player.set_xwindow(self.GetHandle()) # this line messes up windows
			# FIXME: this should be made cross-platform
			self.OnPlay()

			# set the volume slider to the current volume
			#self.volslider.SetValue(self.player.audio_get_volume() / 2)
			self.volslider.set(self.player.audio_get_volume())
	def OnOpen(self):
		"""Pop up a new dialow window to choose a file, then play the selected file.
		"""
		# if a file is already running, then stop it.
		self.OnStop()

		# Create a file dialog opened in the current home directory, where
		# you can display all kind of files, having as title "Choose a file".
		p = pathlib.Path(os.path.expanduser("~"))
		fullname =  askopenfilename(initialdir = p, title = "choose your file",filetypes = (("all files","*.*"),("mp4 files","*.mp4")))
		#fullname = self.OnDetect(fullname)
		if os.path.isfile(fullname):
			dirname  = os.path.dirname(fullname)
			filename = os.path.basename(fullname)
			# Creation
			self.Media = self.Instance.media_new(str(os.path.join(dirname, filename)))
			self.player.set_media(self.Media)
			# Report the title of the file chosen
			#title = self.player.get_title()
			#  if an error was encountred while retriving the title, then use
			#  filename
			#if title == -1:
			#    title = filename
			#self.SetTitle("%s - tkVLCplayer" % title)

			# set the window id where to render VLC's video output
			if platform.system() == 'Windows':
				self.player.set_hwnd(self.GetHandle())
			else:
				self.player.set_xwindow(self.GetHandle()) # this line messes up windows
			# FIXME: this should be made cross-platform
			self.OnPlay()

			# set the volume slider to the current volume
			#self.volslider.SetValue(self.player.audio_get_volume() / 2)
			self.volslider.set(self.player.audio_get_volume())

	def OnPlay(self):
		"""Toggle the status to Play/Pause.
		If no file is loaded, open the dialog window.
		"""
		# check if there is a file to play, otherwise open a
		# if self.player.get_time() == self.player.get_length():
		# 	print("Video reach end")
		# 	self.OnStop()
		# Tk.FileDialog to select a file
		if not self.player.get_media():
			self.OnOpen()
		else:
			# Try to launch the media, if this fails display an error message
			if self.player.play() == -1:
				self.errorDialog("Unable to play.")

	def GetHandle(self):
		return self.videopanel.winfo_id()

	#def OnPause(self, evt):
	def OnPause(self):
		"""Pause the player.
		"""
		self.player.pause()

	def OnStop(self):
		"""Stop the player.
		"""
		self.player.stop()
		# reset the time slider
		self.timeslider.set(0)

	def OnTimer(self):
		"""Update the time slider according to the current movie time.
		"""
		if self.player == None:
			return
		# since the self.player.get_length can change while playing,
		# re-set the timeslider to the correct range.
		length = self.player.get_length()
		dbl = length * 0.001
		self.timeslider.config(to=dbl)

		# update the time on the slider
		tyme = self.player.get_time()
		if tyme == -1:
			tyme = 0
		dbl = tyme * 0.001
		self.timeslider_last_val = ("%.0f" % dbl) + ".0"
		# don't want to programatically change slider while user is messing with it.
		# wait 2 seconds after user lets go of slider
		if time.time() > (self.timeslider_last_update + 2.0):
			self.timeslider.set(dbl)

	def scale_sel(self, evt):
		if self.player == None:
			return
		nval = self.scale_var.get()
		sval = str(nval)
		if self.timeslider_last_val != sval:
			# this is a hack. The timer updates the time slider.
			# This change causes this rtn (the 'slider has changed' rtn) to be invoked.
			# I can't tell the difference between when the user has manually moved the slider and when
			# the timer changed the slider. But when the user moves the slider tkinter only notifies
			# this rtn about once per second and when the slider has quit moving.
			# Also, the tkinter notification value has no fractional seconds.
			# The timer update rtn saves off the last update value (rounded to integer seconds) in timeslider_last_val
			# if the notification time (sval) is the same as the last saved time timeslider_last_val then
			# we know that this notification is due to the timer changing the slider.
			# otherwise the notification is due to the user changing the slider.
			# if the user is changing the slider then I have the timer routine wait for at least
			# 2 seconds before it starts updating the slider again (so the timer doesn't start fighting with the
			# user)
			self.timeslider_last_update = time.time()
			mval = "%.0f" % (nval * 1000)
			self.player.set_time(int(mval)) # expects milliseconds


	def volume_sel(self, evt):
		if self.player == None:
			return
		volume = self.volume_var.get()
		if volume > 100:
			volume = 100
		if self.player.audio_set_volume(volume) == -1:
			self.errorDialog("Failed to set volume")



	def OnToggleVolume(self, evt):
		"""Mute/Unmute according to the audio button.
		"""
		is_mute = self.player.audio_get_mute()

		self.player.audio_set_mute(not is_mute)
		# update the volume slider;
		# since vlc volume range is in [0, 200],
		# and our volume slider has range [0, 100], just divide by 2.
		self.volume_var.set(self.player.audio_get_volume())

	def OnSetVolume(self):
		"""Set the volume according to the volume sider.
		"""
		volume = self.volume_var.get()
		# vlc.MediaPlayer.audio_set_volume returns 0 if success, -1 otherwise
		if volume > 100:
			volume = 100
		if self.player.audio_set_volume(volume) == -1:
			self.errorDialog("Failed to set volume")

	def errorDialog(self, errormessage):
		"""Display a simple error dialog.
		"""
		messagebox.showerror('Error', errormessage)

def Tk_get_root():
	if not hasattr(Tk_get_root, "root"): #(1)
		Tk_get_root.root= Tk.Tk()  #initialization call is inside the function
	return Tk_get_root.root

def _quit():
	print("_quit: bye")
	root = Tk_get_root()
	root.quit()     # stops mainloop
	root.destroy()  # this is necessary on Windows to prevent
					# Fatal Python Error: PyEval_RestoreThread: NULL tstate
	os._exit(1)

if __name__ == "__main__":
	# Create a Tk.App(), which handles the windowing system event loop
	root = Tk_get_root()
	root.protocol("WM_DELETE_WINDOW", _quit)

	player = Player(root, title="Action Recognition Demo")
	# show the player window centred and run the application
	root.mainloop()