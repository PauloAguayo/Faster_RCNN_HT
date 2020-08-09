from __future__ import print_function
import numpy as np
from kalman_tracker import KalmanBoxTracker
from data_association import associate_detections_to_trackers

class Sort:
  def __init__(self,max_age=6,min_hits=1): ## max_age=10 , min_hits??????

    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,detections,img=None):
    """
    Params:
      detections - a numpy array of detections in the format [[ymin,xmin,ymax,xmax],[ymin,xmin,ymax,xmax],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(img) #for kal!
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    if detections != []:
      matched, unmatched_detections, unmatched_trks = associate_detections_to_trackers(detections,trks)
      #update matched trackers with assigned detections
      for t,trk in enumerate(self.trackers):
        if(t not in unmatched_trks):
          d = matched[np.where(matched[:,1]==t)[0],0]
          trk.update(detections[d[0]],img) ## for dlib re-intialize the trackers ?!

      #create and initialise new trackers for unmatched detections
      for i in unmatched_detections:
        trk = KalmanBoxTracker(detections[i])
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        if detections == []:
          trk.update([],img)
        d = trk.get_state()
        if((trk.time_since_update < 3) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)

    if(len(ret)>0):
      return np.concatenate(ret)

    return np.empty((0,5))
