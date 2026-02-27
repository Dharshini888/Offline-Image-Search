import numpy as np
from sklearn.cluster import DBSCAN
import datetime
import logging

logger = logging.getLogger("ClusteringEngine")

class ClusteringEngine:
    def detect_events(self, images_metadata):
        """
        Clusters images into events based on time and GPS coordinates.
        images_metadata: list of dicts with 'lat', 'lon', 'timestamp'.
        """
        if not images_metadata:
            return []

        # Convert timestamps to numeric values (e.g., hours since first photo)
        start_time = min(m['timestamp'] for m in images_metadata)
        
        data = []
        for m in images_metadata:
            # Time feature: hours since start (scaled for DBSCAN)
            t_diff = (m['timestamp'] - start_time).total_seconds() / 3600.0
            
            # GPS features: lat, lon
            # Note: For real clusters, we should normalize these or use haversine metric
            lat = m.get('lat', 0) or 0
            lon = m.get('lon', 0) or 0
            
            # Weighted vector: Time is heavily weighted for "same day" events
            # Tuning: 1 unit of time = 1 hour. eps=12 means 12 hours max gap.
            data.append([t_diff, lat, lon])
            
        X = np.array(data)
        
        # Cluster based on time and location
        # eps and min_samples need tuning based on representative datasets
        model = DBSCAN(eps=24.0, min_samples=2) # 24 hour gap allowed
        labels = model.fit_predict(X)
        
        return labels

clustering_engine = ClusteringEngine()
