
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[3]:


collision_df=pd.read_csv('/Users/yuantaoli/Downloads/NYPD_Motor_Vehicle_Collisions.csv')


# In[4]:


collision_df


# In[5]:


#We only care about new york data
collision_df=collision_df[(collision_df.LONGITUDE>-74.25)&(collision_df.LONGITUDE<-73.75)&(collision_df.LATITUDE<40.9)&(collision_df.LATITUDE>40.5)]


# In[6]:


kill_df=collision_df[(collision_df['NUMBER OF PERSONS KILLED'])>0]


# In[7]:



#longitude = list(collision_df.LONGITUDE)
#latitude=list(collision_df.LATITUDE)
#kill_longitude = list(kill_df.LONGITUDE)
#kill_latitude=list(kill_df.LATITUDE)
#plt.figure(figsize = (10,10))
#plt.plot(longitude,latitude,'.', alpha=0.4, markersize=0.05, label='Acidents')
#plt.plot(kill_longitude,kill_latitude,'.', color='r', alpha=1, markersize=2, label='Deadly accidents')
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.title('Car accidents and Deadly car accidents')
#plt.legend()
#plt.savefig("figure04.png")
#plt.show()


# In[8]:


#collision_df_loc=collision_df[['LATITUDE','LONGITUDE']]
#kmeans=KMeans(n_clusters=25, random_state=5, n_init=10).fit(collision_df_loc)
#collision_df_loc['labels']=kmeans.labels_


# In[9]:


#collision_df_loc


# In[10]:


#plt.figure(figsize = (10,10))
#for label in collision_df_loc.labels.unique():
#    plt.plot(collision_df_loc.LONGITUDE[collision_df_loc.labels == label],collision_df_loc.LATITUDE[collision_df_loc.labels == label],'.', alpha = 0.3, markersize = 0.3)

#plt.title('Clusters of New York')
#plt.show()
#plt.savefig("figure03.png")


# In[7]:


collision_df['TIME']=pd.to_datetime(collision_df['TIME']).apply(lambda x:x.time())


# In[8]:


workhours=collision_df[(collision_df['TIME']>pd.to_datetime('09:00:00').time())&(collision_df['TIME']<pd.to_datetime('17:00:00').time())]


# In[9]:


offhours=collision_df[(collision_df['TIME']<pd.to_datetime('09:00:00').time())|(collision_df['TIME']>pd.to_datetime('17:00:00').time())]


# In[14]:


#offhours


# In[15]:


#work_longitude = list(workhours.LONGITUDE)
#work_latitude=list(workhours.LATITUDE)
#off_longitude=list(offhours.LONGITUDE)
#off_latitude=list(offhours.LATITUDE)

#plt.figure(figsize = (10,10))
#plt.plot(work_longitude,work_latitude,'.', color='r' ,alpha=0.4, markersize=0.05,label='9:00-5:00')
#plt.plot(off_longitude,off_latitude,'.',alpha=0.4, markersize=0.05,label='off work hours')
#plt.title('Work hour accidents and off hour accidents')
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.legend()
#plt.savefig("figure01.png")
#plt.show()


# In[16]:


#kill_df


# In[12]:


import geopandas as gpd


# In[ ]:


#load the 2018 september data
sept_18 = pd.read_csv('/Users/yuantaoli/Downloads/traffic data/september2018.csv')


# In[19]:


sept_18


# In[20]:


#linkinfo=pd.read_csv('/Users/yuantaoli/Downloads/traffic data/linkinfo.csv')


# In[21]:


#sept_18.Id.unique()


# In[22]:


#linkinfo_short=linkinfo[['linkId','linkPoints']]


# In[23]:


#sept_18m=pd.merge(sept_18,linkinfo_short,how='inner',on='linkId')


# In[24]:


#sept_18m['Latitude']=sept_18m.linkPoints.str.split('[ ,]+',n=2,expand=True)[0]


# In[25]:


#Sept_18m is the Sept data with location of cameras included
#sept_18m['Longitude']=sept_18m.linkPoints.str.split('[ ,]+',n=2,expand=True)[1]


# In[26]:


#sept_18m['Latitude']=sept_18m['Latitude'].astype(float)
#sept_18m['Longitude']=sept_18m['Longitude'].astype(float)


# In[27]:


#longitude = list(collision_df.LONGITUDE)
#latitude=list(collision_df.LATITUDE)
#kill_longitude = list(kill_df.LONGITUDE)
#kill_latitude=list(kill_df.LATITUDE)
#camera_longitude = list(sept_18m['Longitude'])
#camera_latitude=list(sept_18m['Latitude'])
#plt.figure(figsize = (10,10))
#plt.plot(longitude,latitude,'.', alpha=0.4, markersize=0.05, label='Acidents')
#plt.plot(kill_longitude,kill_latitude,'.', color='r', alpha=1, markersize=2, label='Deadly accidents')
#plt.plot(camera_longitude,camera_latitude,'o', color='b', alpha=1, markersize=2, label='cameras')
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.title('Car accidents and Deadly car accidents')
#plt.legend()
#plt.savefig("figure0.png")
#plt.show()


# In[ ]:


collision_df['DATE']=pd.to_datetime(collision_df['DATE'])


# In[29]:


#kill_df


# In[ ]:


kill_df['DATE']=pd.to_datetime(kill_df['DATE'])


# In[31]:


#collision_df_1809=collision_df[(collision_df['DATE']>=pd.to_datetime('2018-09-01'))&(collision_df['DATE']<=pd.to_datetime('2018-09-30'))]


# In[32]:


#kill_1809=kill_df[(kill_df['DATE']>=pd.to_datetime('2018-09-01'))&(kill_df['DATE']<=pd.to_datetime('2018-09-30'))]


# In[33]:


#collision_df_1809


# In[34]:


#longitude = list(collision_df_1809.LONGITUDE)
#latitude=list(collision_df_1809.LATITUDE)
#kill_longitude = list(kill_1809.LONGITUDE)
#kill_latitude=list(kill_1809.LATITUDE)
#camera_longitude = list(sept_18m['Longitude'])
#camera_latitude=list(sept_18m['Latitude'])
#plt.figure(figsize = (10,10))
#plt.plot(longitude,latitude,'.', alpha=0.4, markersize=0.05, label='Acidents')
#plt.plot(kill_longitude,kill_latitude,'.', color='r', alpha=1, markersize=2, label='Deadly accidents')
#plt.plot(camera_longitude,camera_latitude,'o', color='b', alpha=1, markersize=2, label='cameras')
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.title('Car accidents and Deadly car accidents')
#plt.legend()
#plt.savefig("figure1.png")
#plt.show()


# In[35]:


#sept_18m[sept_18m['Speed']>100]


# In[36]:


#sept_18m['Speed'].min()


# In[37]:


#sept_18m['Speed_Group']=pd.cut(sept_18m['Speed'],10)


# In[38]:


#sept_18m['Speed_Group']


# In[39]:


#sept_18m.groupby('Speed_Group').count()


# In[40]:


#sept_18m_60=sept_18m[sept_18m['Speed']>=60]


# In[41]:


import geopandas as gpd
#gdf = gpd.read_file('/Users/yuantaoli/Downloads/traffic data/speed_limit_shapefile/speed_limit_shapefile.shp')


# In[42]:


#%matplotlib inline
#figsize=(10,10)
#gdf.plot()


# In[43]:


#len(gdf)


# In[44]:


#gdf['geometry'].astype(str)[0]


# In[ ]:


import requests
import json
from pandas.io.json import json_normalize


# In[46]:


#with open('/Users/yuantaoli/Downloads/traffic data/speed_limit_manhattan.txt','r') as json_file:
#     mht_data=json.load(json_file)
#mht_limit = pd.DataFrame.from_dict(json_normalize(mht_data['features']), orient='columns')
#mht_limit.loc[:,'Longitude']=mht_limit['geometry.coordinates'].map(lambda x: x[0][0])
#mht_limit.loc[:,'Longitude_end']=mht_limit['geometry.coordinates'].map(lambda x: x[1][0])
#mht_limit.loc[:,'Latitude']=mht_limit['geometry.coordinates'].map(lambda x: x[0][1])
#mht_limit.loc[:,'Latitude_end']=mht_limit['geometry.coordinates'].map(lambda x: x[1][1])


# In[47]:


#with open('/Users/yuantaoli/Downloads/traffic data/speed_limit_brooklyn.json','r') as json_file:
#     bkln_data=json.load(json_file)
#bkln_limit = pd.DataFrame.from_dict(json_normalize(bkln_data['features']), orient='columns')
#bkln_limit.loc[:,'Longitude']=bkln_limit['geometry.coordinates'].map(lambda x: x[0][0])
#bkln_limit.loc[:,'Longitude_end']=bkln_limit['geometry.coordinates'].map(lambda x: x[1][0])
#bkln_limit.loc[:,'Latitude']=bkln_limit['geometry.coordinates'].map(lambda x: x[0][1])
#bkln_limit.loc[:,'Latitude_end']=bkln_limit['geometry.coordinates'].map(lambda x: x[1][1])


# In[48]:


#with open('/Users/yuantaoli/Downloads/traffic data/speed_limit_bronx.json','r') as json_file:
#     bx_data=json.load(json_file)
#bx_limit = pd.DataFrame.from_dict(json_normalize(bx_data['features']), orient='columns')
#bx_limit.loc[:,'Longitude']=bx_limit['geometry.coordinates'].map(lambda x: x[0][0])
#bx_limit.loc[:,'Longitude_end']=bx_limit['geometry.coordinates'].map(lambda x: x[1][0])
#bx_limit.loc[:,'Latitude']=bx_limit['geometry.coordinates'].map(lambda x: x[0][1])
#bx_limit.loc[:,'Latitude_end']=bx_limit['geometry.coordinates'].map(lambda x: x[1][1])


# In[49]:


#with open('/Users/yuantaoli/Downloads/traffic data/speed_limit_queens.json','r') as json_file:
#     qn_data=json.load(json_file)
#qn_limit = pd.DataFrame.from_dict(json_normalize(qn_data['features']), orient='columns')
#qn_limit.loc[:,'Longitude']=qn_limit['geometry.coordinates'].map(lambda x: x[0][0])
#qn_limit.loc[:,'Longitude_end']=qn_limit['geometry.coordinates'].map(lambda x: x[1][0])
#qn_limit.loc[:,'Latitude']=qn_limit['geometry.coordinates'].map(lambda x: x[0][1])
#qn_limit.loc[:,'Latitude_end']=qn_limit['geometry.coordinates'].map(lambda x: x[1][1])


# In[50]:


#with open('/Users/yuantaoli/Downloads/traffic data/speed_limit_statenisland.json','r') as json_file:
#     st_data=json.load(json_file)
#st_limit = pd.DataFrame.from_dict(json_normalize(st_data['features']), orient='columns')
#st_limit.loc[:,'Longitude']=st_limit['geometry.coordinates'].map(lambda x: x[0][0])
#st_limit.loc[:,'Longitude_end']=st_limit['geometry.coordinates'].map(lambda x: x[1][0])
#st_limit.loc[:,'Latitude']=st_limit['geometry.coordinates'].map(lambda x: x[0][1])
#st_limit.loc[:,'Latitude_end']=st_limit['geometry.coordinates'].map(lambda x: x[1][1])


# In[51]:


#bx_limit.head(1)


# In[52]:


#mht_limit.head(1)


# In[53]:


#qn_limit.head(1)


# In[54]:


#bkln_limit.head(1)


# In[55]:


#st_limit.head(1)


# In[56]:


#speed_limit=pd.concat([mht_limit,qn_limit,bx_limit,bkln_limit,st_limit])


# In[57]:


#speed_limit


# In[58]:


#speed_limit['properties.postvz_sl'].min()


# In[59]:


#speed_limit20=speed_limit[speed_limit['properties.postvz_sl']==20]
#speed_limit25=speed_limit[speed_limit['properties.postvz_sl']==25]
#speed_limit30=speed_limit[speed_limit['properties.postvz_sl']==30]
#speed_limit35=speed_limit[speed_limit['properties.postvz_sl']==35]
#speed_limit40=speed_limit[speed_limit['properties.postvz_sl']==40]
#speed_limit45=speed_limit[speed_limit['properties.postvz_sl']==45]
#speed_limit50=speed_limit[speed_limit['properties.postvz_sl']==50]


# In[60]:


#speed_limit50['geometry.coordinates'][0]


# In[61]:


#plt.figure(figsize = (10,10))
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)


#plt.plot([-73.98989832599995,-73.98943021599996],[40.77790445900007,40.77854581200006],'k-')


# In[62]:


#speed_limit.groupby('properties.postvz_sl').count()


# In[63]:


#speed_ensemble=[speed_limit20,speed_limit25,speed_limit30,speed_limit35,speed_limit40,speed_limit45,speed_limit50]


# In[64]:


#plt.figure(figsize = (10,10))
#for spl in speed_ensemble:
#    speedlimit_longitude=spl['Longitude'].tolist()
#    speedlimit_latitude=spl['Latitude'].tolist()
#    speedlimit_longitude_end=spl['Longitude_end'].tolist()
#    speedlimit_latitude_end=spl['Latitude_end'].tolist()
#    plt.plot(zip(speedlimit_longitude,speedlimit_longitude_end),zip(speedlimit_latitude,speedlimit_latitude_end))

#longitude = collision_df.LONGITUDE.tolist()
#latitude=collision_df.LATITUDE.tolist()
#kill_longitude = kill_df.LONGITUDE.tolist()
#kill_latitude=kill_df.LATITUDE.tolist()
#plt.plot(longitude,latitude,'.', alpha=0.4, markersize=0.05, label='Acidents')
#plt.plot(kill_longitude,kill_latitude,'.', color='r', alpha=1, markersize=2, label='Deadly accidents')

#plt.ylim(40.5,40.9)
#plt.legend()
#plt.xlim(-74.25,-73.75)
#plt.title('Speed limit')
#plt.show()


# In[65]:


#speedlimit_longitude25=speed_limit25['Longitude'].tolist()
#speedlimit_latitude25=speed_limit25['Latitude'].tolist()
#speedlimit_longitude_end25=speed_limit25['Longitude_end'].tolist()
#speedlimit_latitude_end25=speed_limit25['Latitude_end'].tolist()
#plt.figure(figsize = (10,10))
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#for i in range(len(speedlimit_longitude25)):
#    plt.plot([speedlimit_longitude25[i],speedlimit_longitude_end25[i]],[speedlimit_latitude25[i],speedlimit_latitude_end25[i]],'k-', color='yellow')
#speedlimit_longitude50=speed_limit50['Longitude'].tolist()
#speedlimit_latitude50=speed_limit50['Latitude'].tolist()
#speedlimit_longitude_end50=speed_limit50['Longitude_end'].tolist()
#speedlimit_latitude_end50=speed_limit50['Latitude_end'].tolist()
#for i in range(len(speedlimit_longitude50)):
#    plt.plot([speedlimit_longitude50[i],speedlimit_longitude_end50[i]],[speedlimit_latitude50[i],speedlimit_latitude_end50[i]],'k-', color='red')
#speedlimit_longitude20=speed_limit20['Longitude'].tolist()
#speedlimit_latitude20=speed_limit20['Latitude'].tolist()
#speedlimit_longitude_end20=speed_limit20['Longitude_end'].tolist()
#speedlimit_latitude_end20=speed_limit20['Latitude_end'].tolist()
#for i in range(len(speedlimit_longitude20)):
#    plt.plot([speedlimit_longitude20[i],speedlimit_longitude_end20[i]],[speedlimit_latitude20[i],speedlimit_latitude_end20[i]],'k-', color='blue')
#camera_longitude =sept_18m['Longitude'].tolist()
#camera_latitude=sept_18m['Latitude'].tolist()
#plt.plot(camera_longitude,camera_latitude,'o', color='black', alpha=1, markersize=2, label='cameras')
#plt.title('speed limit and location of cameras')
#plt.savefig("figure14.png")


# In[66]:


#speedlimit_longitude50=speed_limit50['Longitude'].tolist()
#speedlimit_latitude50=speed_limit50['Latitude'].tolist()
#speedlimit_longitude_end50=speed_limit50['Longitude_end'].tolist()
#speedlimit_latitude_end50=speed_limit50['Latitude_end'].tolist()
#plt.figure(figsize = (10,10))
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#for i in range(len(speedlimit_longitude50)):
#    plt.plot([speedlimit_longitude50[i],speedlimit_longitude_end50[i]],[speedlimit_latitude50[i],speedlimit_latitude_end50[i]],'k-')
#plt.legend('Speed limit 50 mph')


# In[67]:


#speedlimit_longitude20=speed_limit20['Longitude'].tolist()
#speedlimit_latitude20=speed_limit20['Latitude'].tolist()
#speedlimit_longitude_end20=speed_limit20['Longitude_end'].tolist()
#speedlimit_latitude_end=speed_limit20['Latitude_end'].tolist()
#plt.figure(figsize = (10,10))
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#for i in range(len(speedlimit_longitude)):
#    plt.plot([speedlimit_longitude20[i],speedlimit_longitude_end20[i]],[speedlimit_latitude20[i],speedlimit_latitude_end20[i]],'k-')
#plt.legend('Speed limit 20 mph')


# In[68]:


#plt.figure(figsize = (10,10))

#speedlimit_longitude=speed_limit25['Longitude'].tolist()
#speedlimit_latitude=speed_limit25['Latitude'].tolist()
#plt.plot(speedlimit_longitude,speedlimit_latitude,'.', alpha=0.4, markersize=2)

#longitude = collision_df.LONGITUDE.tolist()
#latitude=collision_df.LATITUDE.tolist()
#kill_longitude = kill_df.LONGITUDE.tolist()
#kill_latitude=kill_df.LATITUDE.tolist()
#plt.plot(longitude,latitude,'.', alpha=0.4, markersize=0.05, label='Acidents')
#plt.plot(kill_longitude,kill_latitude,'.', color='r', alpha=1, markersize=2, label='Deadly accidents')

#plt.ylim(40.5,40.9)
#plt.legend()
#plt.xlim(-74.25,-73.75)
#plt.title('Speed limit')
#plt.show()


# In[70]:


#Draw a 2D heat map and see wherever the city has the most accident
collision_df_plot=collision_df[(collision_df.LONGITUDE>-74.25)&(collision_df.LONGITUDE<-73.75)&(collision_df.LATITUDE<40.9)&(collision_df.LATITUDE>40.5)]
lon=collision_df_plot.LONGITUDE
lon=lon.dropna()
lat=collision_df_plot.LATITUDE
lat=lat.dropna()
heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.figure(figsize = (15,15))
plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=100., cmap=plt.cm.inferno)
plt.colorbar(fraction=0.046, pad=0.04)
plt.ylim(40.5,40.9)
plt.xlim(-74.25,-73.75)
plt.title('Collisions')
plt.show()
plt.savefig("figure12.png")


# In[ ]:


collision_df=collision_df.dropna(subset={'TIME'})


# In[ ]:


collision_df['HOUR']=collision_df['TIME'].apply(lambda x: x.hour)


# In[73]:


#collision_df.groupby('HOUR').count()['DATE'].plot(kind='bar')
#plt.title('Count of Collisions vs Hours')
#plt.savefig("figure11.png")


# In[ ]:


kill_df=collision_df[(collision_df['NUMBER OF PERSONS KILLED'])>0]


# In[ ]:


kill_df=kill_df.dropna(subset={'TIME'})
kill_df['HOUR']=kill_df['TIME'].apply(lambda x: x.hour)
#kill_df.groupby('HOUR').count()['DATE'].plot(kind='bar')
#plt.title('Count of Deaths vs Hours')
#plt.savefig("figure10.png")


# In[ ]:


collision_hours=pd.DataFrame(collision_df.groupby('HOUR').sum())
collision_hours=collision_hours.reset_index()
collision_counts=pd.DataFrame(collision_df.groupby('HOUR').count()['DATE'])
collision_counts=collision_counts.reset_index()


# In[ ]:


collision_hours=pd.merge(collision_hours,collision_counts, how='outer')


# In[ ]:


collision_hours=collision_hours.rename(columns={'DATE':'COUNT'})
collision_hours['injury_rate']=collision_hours['NUMBER OF PERSONS INJURED']/collision_hours['COUNT']
collision_hours['death_rate']=collision_hours['NUMBER OF PERSONS KILLED']/collision_hours['COUNT']


# In[ ]:


fig,axs=plt.subplots(1,1, figsize=(10,5))
axs.bar(collision_hours['HOUR'],collision_hours['COUNT'],alpha=0.6,color='coral',label='collision counts')
axs.legend(loc='upper left')
axs.set_ylabel('collision counts')
axs.set_xlabel('hours')
axs2 = axs.twinx()
axs2.bar(collision_hours['HOUR'],collision_hours['injury_rate'],alpha=0.6,color='green',label='injury rate')
axs2.bar(collision_hours['HOUR'],10*collision_hours['death_rate'],alpha=0.6,color='black',label='10x death rate')
axs2.legend(loc='upper right')
axs2.set_ylim(0,0.45)
axs.set_title('Hours, collisions, deaths and injuries')


# In[80]:


#kill_df_plot=kill_df[(kill_df.LONGITUDE>-74.25)&(kill_df.LONGITUDE<-73.75)&(kill_df.LATITUDE<40.9)&(kill_df.LATITUDE>40.5)]
#lon=kill_df_plot.LONGITUDE
#lon=lon.dropna()
#lat=kill_df_plot.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=250)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#plt.figure(figsize = (15,15))
#plt.clf()
#plt.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=4., cmap=plt.cm.inferno)
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.title('Kills')
#plt.show()
#plt.savefig("figure9.png")


# In[ ]:


kill_df.groupby('CONTRIBUTING FACTOR VEHICLE 1').sum()


# In[11]:


collision_df['CONTRIBUTING FACTOR VEHICLE 1'].unique()


# In[83]:


collision_df.loc[collision_df['CONTRIBUTING FACTOR VEHICLE 1']=='Illnes','CONTRIBUTING FACTOR VEHICLE 1']='Illness'


# In[84]:


collision_df.loc[collision_df['CONTRIBUTING FACTOR VEHICLE 1']=='Drugs (illegal)','CONTRIBUTING FACTOR VEHICLE 1']='Drugs (Illegal)'


# In[85]:


collision_df


# In[86]:


casualty_sum=pd.DataFrame(collision_df.groupby('CONTRIBUTING FACTOR VEHICLE 1')[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED','NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED']].sum().sort_values('NUMBER OF PERSONS INJURED',ascending=False))


# In[87]:


casualty_count=pd.DataFrame(collision_df.groupby('CONTRIBUTING FACTOR VEHICLE 1').count()['DATE'])


# In[88]:


casualty_count=casualty_count.rename(columns={'DATE':'Count'})
casualty_count=pd.DataFrame(casualty_count['Count'])


# In[89]:


casualty_sum=casualty_sum.reset_index()
casualty_count=casualty_count.reset_index()


# In[90]:


casualty_sum=pd.merge(casualty_count,casualty_sum)


# In[91]:


casualty_sum.sort_values(by='Count',ascending=False)


# In[92]:


#Casualty_sum includes data of casualties and those that did not result in any casualty
casualty_sum['Injury_ratio']=casualty_sum['NUMBER OF PERSONS INJURED']/casualty_sum['Count']


# In[93]:


casualty_sum['death_ratio']=casualty_sum['NUMBER OF PERSONS KILLED']/casualty_sum['Count']


# In[94]:


casualty_sum.sort_values('Injury_ratio',ascending=False)


# In[95]:


casualty_sum.sort_values('NUMBER OF PERSONS INJURED',ascending=False)


# In[96]:


casualty_sum.sort_values('NUMBER OF PERSONS KILLED',ascending=False)


# In[97]:


#casualty_sum['NUMBER OF PERSONS KILLED'].sum()


# In[98]:


#casualty_sum['NUMBER OF PERSONS INJURED'].sum()


# In[99]:


#collision_df.to_csv('collision_df_fixed.csv')
#casualty_sum.to_csv('casualty_sum.csv')


# In[100]:


disregard_control=collision_df[collision_df['CONTRIBUTING FACTOR VEHICLE 1']=='Traffic Control Disregarded']


# In[101]:


#longitude = collision_df.LONGITUDE.tolist()
#latitude = collision_df.LATITUDE.tolist()
#longitude_dc=disregard_control.LONGITUDE.tolist()
#latitude_dc=disregard_control.LATITUDE.tolist()
#plt.figure(figsize = (10,10))
#plt.plot(longitude,latitude,'.', alpha=0.4, markersize=0.5, label='Acidents')
#plt.plot(longitude_dc,latitude_dc,'o', color='red', alpha=1, markersize=2, label='cameras')
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.title('traffic control disregard')
#plt.legend()
#plt.savefig("figure2.png")
#plt.show()


# In[102]:


#disregard_control.LONGITUDE.describe()


# In[103]:


from PIL import Image
map_NYC = Image.open("/Users/yuantaoli/Downloads/traffic data/map.png")
BB = (-74.25, -73.75, 40.507, 40.9)


# In[106]:


disregard_control=disregard_control[(disregard_control.LONGITUDE>-74.25)&(disregard_control.LONGITUDE<-73.75)&(disregard_control.LATITUDE<40.9)&(disregard_control.LATITUDE>40.5)]
#lon=disregard_control.LONGITUDE
#lon=lon.dropna()
#lat=disregard_control.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#plt.figure(figsize = (25,25))
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=5., cmap=plt.cm.inferno, zorder=1, alpha=0.6)
#plt.imshow(map_NYC,zorder=0, extent=BB);
#plt.colorbar(fraction=0.046, pad=0.04)
#plt.title('disregard traffic control', fontsize=20)
#plt.savefig("figure3.png")
#plt.show()


# In[108]:


disregard_summary=pd.DataFrame(disregard_control.groupby('HOUR').sum()[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']])
disregard_summary=disregard_summary.reset_index()
disregard_count=pd.DataFrame(disregard_control.groupby('HOUR').count()['DATE'])
disregard_count=disregard_count.rename(columns={'DATE':'COUNT'})
disregard_count=disregard_count.reset_index()
disregard_summary=pd.merge(disregard_summary,disregard_count)
disregard_summary['injury_rate']=disregard_summary['NUMBER OF PERSONS INJURED']/disregard_summary['COUNT']
disregard_summary['death_rate']=disregard_summary['NUMBER OF PERSONS KILLED']/disregard_summary['COUNT']


# In[109]:


#fig,axs=plt.subplots(1,1, figsize=(10,5))
#axs.bar(disregard_summary['HOUR'],disregard_summary['COUNT'],alpha=0.6,color='coral',label='collision counts')
#axs.legend(loc='upper left')
#axs.set_ylabel('collision counts')
#axs.set_xlabel('hours')
#axs2 = axs.twinx()
#axs2.bar(disregard_summary['HOUR'],disregard_summary['injury_rate'],alpha=0.6,color='green',label='injury rate')
#axs2.bar(disregard_summary['HOUR'],10*disregard_summary['death_rate'],alpha=0.6,color='black',label='10x death rate')
#axs2.legend(loc='upper right')
#axs2.set_ylim(0,0.8)
#axs.set_title('Disregarding traffic control')
#plt.savefig("figure8.png")


# In[110]:


ped_cyl_etc=collision_df[collision_df['CONTRIBUTING FACTOR VEHICLE 1']=='Pedestrian/Bicyclist/Other Pedestrian Error/Confusion']


# In[111]:


#lon=ped_cyl_etc.LONGITUDE
#lon=lon.dropna()
#lat=ped_cyl_etc.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=100)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#plt.figure(figsize = (25,25))
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=10., cmap=plt.cm.inferno, zorder=1, alpha=0.6)
#plt.imshow(map_NYC,zorder=0, extent=BB);
#plt.colorbar(fraction=0.046, pad=0.04)
#plt.title('peds cyclists and other people error', fontsize=20)
#plt.savefig("figure4.png")
#plt.show()


# In[112]:


ped_cyl_etc_summary=pd.DataFrame(ped_cyl_etc.groupby('HOUR').sum()[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']])
ped_cyl_etc_summary=ped_cyl_etc_summary.reset_index()
ped_cyl_etc_count=pd.DataFrame(ped_cyl_etc.groupby('HOUR').count()['DATE'])
ped_cyl_etc_count=ped_cyl_etc_count.rename(columns={'DATE':'COUNT'})
ped_cyl_etc_count=ped_cyl_etc_count.reset_index()
ped_cyl_etc_summary=pd.merge(ped_cyl_etc_summary,disregard_count)
ped_cyl_etc_summary['injury_rate']=ped_cyl_etc_summary['NUMBER OF PERSONS INJURED']/ped_cyl_etc_summary['COUNT']
ped_cyl_etc_summary['death_rate']=ped_cyl_etc_summary['NUMBER OF PERSONS KILLED']/ped_cyl_etc_summary['COUNT']


# In[113]:


#fig,axs=plt.subplots(1,1, figsize=(10,5))
#axs.bar(ped_cyl_etc_summary['HOUR'],ped_cyl_etc_summary['COUNT'],alpha=0.6,color='coral',label='collision counts')
#axs.legend(loc='upper left')
#axs.set_ylabel('collision counts')
#axs.set_xlabel('hours')
#axs2 = axs.twinx()
#axs2.bar(ped_cyl_etc_summary['HOUR'],ped_cyl_etc_summary['injury_rate'],alpha=0.6,color='green',label='injury rate')
#axs2.bar(ped_cyl_etc_summary['HOUR'],10*ped_cyl_etc_summary['death_rate'],alpha=0.6,color='black',label='10x death rate')
#axs2.legend(loc='upper right')
#axs2.set_ylim(0,0.8)
#axs.set_title('Pesdestrians/cyclists errors')
#plt.savefig("figure5.png")


# In[114]:


#collision_df['CONTRIBUTING FACTOR VEHICLE 1'].unique()


# In[115]:


#collision_df.to_csv('collision_df.csv')


# In[116]:


Unspecified_acc=collision_df[collision_df['CONTRIBUTING FACTOR VEHICLE 1']=='Unspecified']


# In[117]:


#lon=Unspecified_acc.LONGITUDE
#lon=lon.dropna()
#lat=Unspecified_acc.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#plt.figure(figsize = (25,25))
#plt.ylim(40.5,40.9)
#plt.xlim(-74.25,-73.75)
#plt.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=50., cmap=plt.cm.inferno, zorder=1, alpha=0.6)
#plt.imshow(map_NYC,zorder=0, extent=BB);
#plt.title('Unspecified', fontsize=20)
#plt.savefig("figure6.png")
#plt.show()


# In[118]:


Unspecified_acc=Unspecified_acc.dropna(subset={'TIME'})


# In[119]:


#Unspecified_acc.groupby('HOUR').count()['DATE'].plot(kind='bar')
#plt.title('Count of Unspecified Accidents vs Hours')
#plt.savefig("figure7.png")


# In[120]:


Street_ass=gpd.read_file('/Users/yuantaoli/Downloads/traffic data/StreetAssessmentRating/StreetAssessmentRating.dbf')


# In[121]:


Street_ass=Street_ass.to_crs(epsg=4326)


# In[122]:


Street_ass=Street_ass.sort_values('Rating_B')


# In[123]:


from matplotlib.pyplot import cm


# In[124]:


#fig, axs=plt.subplots(1,figsize=(15,15))
#color=iter(cm.rainbow(np.linspace(0,1,10)))
#for item in Street_ass['Rating_B'].unique():     
#    c=next(color)
#    axs = Street_ass[Street_ass['Rating_B']==item]['geometry'].plot(ax=axs, label='street rating is {}'.format(item),color=c)
#    plt.legend(loc='upper left')
#axs.set_ylim(40.5,40.9)
#axs.set_xlim(-74.25,-73.75)
#axs.set_title('Street rating, 0 means no data')
#fig.savefig("figure21.png")


# In[125]:


#lon=collision_df.LONGITUDE
#lon=lon.dropna()
#lat=collision_df.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#fig, axs=plt.subplots(1,figsize=(15,15))
#axs.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=100., cmap=plt.cm.plasma, alpha=0.5, zorder=1)
#color=iter(cm.rainbow(np.linspace(0,1,10)))
#for item in Street_ass['Rating_B'].unique():     
#    c=next(color)
#    axs = Street_ass[Street_ass['Rating_B']==item]['geometry'].plot(ax=axs, label='street rating is {}'.format(item),color=c, zorder=0)
#    plt.legend(loc='upper left')
#axs.set_ylim(40.5,40.9)
#axs.set_xlim(-74.25,-73.75)
#axs.set_title('Street rating, 0 means no data')
#fig.savefig("figure22.png")


# In[126]:


speedlimit=gpd.read_file('/Users/yuantaoli/Downloads/traffic data/speed_limit_shapefile/speed_limit_shapefile.dbf')


# In[127]:


speedlimit=speedlimit.to_crs(epsg=4326)


# In[128]:


speedlimit=speedlimit.sort_values('postvz_sl')


# In[129]:


#fig, axs=plt.subplots(1,figsize=(15,15))
#color=iter(cm.Reds(np.linspace(0,1,10)))
#for item in speedlimit['postvz_sl'].unique():     
#    c=next(color)
#    axs = speedlimit[speedlimit['postvz_sl']==item]['geometry'].plot(ax=axs, label='speed limit is {}'.format(item),color=c,zorder=0)
#    plt.legend(loc='upper left')
#axs.set_ylim(40.5,40.9)
#axs.set_xlim(-74.25,-73.75)
#axs.set_title('Speed limit')
#fig.savefig("figure23.png")


# In[130]:


#fig, axs=plt.subplots(1,figsize=(15,15))
#lon=collision_df.LONGITUDE
#lon=lon.dropna()
#lat=collision_df.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#axs.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=100., cmap=plt.cm.plasma, alpha=0.8, zorder=1)
#color=iter(cm.Reds(np.linspace(0,1,10)))
#for item in speedlimit['postvz_sl'].unique():     
#    c=next(color)
#    axs = speedlimit[speedlimit['postvz_sl']==item]['geometry'].plot(ax=axs, label='speed limit is {}'.format(item),color=c,zorder=0)
#    plt.legend(loc='upper left')
#axs.set_ylim(40.5,40.9)
#axs.set_xlim(-74.25,-73.75)
#axs.set_title('Speed limit')
#fig.savefig("figure24.png")


# In[131]:


Unsafe_speed=collision_df[collision_df['CONTRIBUTING FACTOR VEHICLE 1']=='Unsafe Speed']


# In[132]:


#fig, axs=plt.subplots(1,figsize=(15,15))
#lon=Unsafe_speed.LONGITUDE
#lon=lon.dropna()
#lat=Unsafe_speed.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#axs.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=2., cmap=plt.cm.plasma, alpha=0.8, zorder=1)
#plt.title('Unsafe speed')


# In[133]:


Unsafe_speed_summary=pd.DataFrame(Unsafe_speed.groupby('HOUR').sum()[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']])
Unsafe_speed_summary=Unsafe_speed_summary.reset_index()
Unsafe_speed_count=pd.DataFrame(Unsafe_speed.groupby('HOUR').count()['DATE'])
Unsafe_speed_count=Unsafe_speed_count.rename(columns={'DATE':'COUNT'})
Unsafe_speed_count=Unsafe_speed_count.reset_index()
Unsafe_speed_summary=pd.merge(Unsafe_speed_summary,disregard_count)
Unsafe_speed_summary['injury_rate']=Unsafe_speed_summary['NUMBER OF PERSONS INJURED']/Unsafe_speed_summary['COUNT']
Unsafe_speed_summary['death_rate']=Unsafe_speed_summary['NUMBER OF PERSONS KILLED']/Unsafe_speed_summary['COUNT']
#fig,axs=plt.subplots(1,1, figsize=(10,5))
#axs.bar(Unsafe_speed_summary['HOUR'],Unsafe_speed_summary['COUNT'],alpha=0.6,color='coral',label='collision counts')
#axs.legend(loc='upper left')
#axs.set_ylabel('collision counts')
#axs.set_xlabel('hours')
#axs2 = axs.twinx()
#axs2.bar(Unsafe_speed_summary['HOUR'],Unsafe_speed_summary['injury_rate'],alpha=0.6,color='green',label='injury rate')
#axs2.bar(Unsafe_speed_summary['HOUR'],10*Unsafe_speed_summary['death_rate'],alpha=0.6,color='black',label='10x death rate')
#axs2.legend(loc='upper right')
#axs2.set_ylim(0,0.8)
#axs.set_title('Speed related accidents')
#plt.savefig("figure25.png")


# In[134]:


#Now it seems like late night is where you do not want to head out
early_morning=collision_df[(collision_df['HOUR']>=4)&(collision_df['HOUR']<=6)]


# In[135]:


early_morning_injury=early_morning[early_morning['NUMBER OF PERSONS INJURED']>0]
early_morning_kill=early_morning[early_morning['NUMBER OF PERSONS KILLED']>0]


# In[136]:


#early_morning


# In[137]:


#fig, axs=plt.subplots(1,figsize=(15,15))
#lon=early_morning.LONGITUDE
#lon=lon.dropna()
#lat=early_morning.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#axs.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=10., cmap=plt.cm.plasma, alpha=0.8, zorder=1)
#axs.set_title('Early Morning Accidents')
#plt.savefig("figure_early_morning.png")


# In[138]:


#fig, axs=plt.subplots(1,figsize=(15,15))
#lon=early_morning_injury.LONGITUDE
#lon=lon.dropna()
#lat=early_morning_injury.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#axs.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=5., cmap=plt.cm.plasma, alpha=0.8, zorder=1)
#axs.set_title('Early Morning Injuries')
#plt.savefig("figure_early_morning_injury.png")


# In[139]:


#fig, axs=plt.subplots(1,figsize=(15,15))
#lon=early_morning_kill.LONGITUDE
#lon=lon.dropna()
#lat=early_morning_kill.LATITUDE
#lat=lat.dropna()
#heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=500)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#axs.imshow(heatmap.T, extent=extent, origin='lower',vmin=0., vmax=1., cmap=plt.cm.plasma, alpha=0.8, zorder=1)
#axs.set_title('Early Morning Kills')
#axs.set_ylim(40.5,40.9)
#axs.set_xlim(-74.25,-73.75)
#plt.imshow(map_NYC,zorder=0, extent=BB);
#plt.savefig("figure_early_morning_death.png")




# In[140]:


collision_temp=collision_df[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED','NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED','CONTRIBUTING FACTOR VEHICLE 1','HOUR']]


# In[141]:


#Pick Canal Street as an example
Canal=speedlimit[(speedlimit['street']=='CANAL STREET')]


# In[142]:


Canal=Canal[Canal['postvz_sg']=='YES']


# In[143]:


Canal=Canal.drop(Canal.index[-3:])


# In[144]:


fig,axs=plt.subplots(1,figsize=(15,15))
axs=Canal['geometry'].plot(zorder=1, alpha=1,ax=axs,color='black')
axs.set_ylim(40.5,40.9)
axs.set_xlim(-74.25,-73.75)
axs.imshow(map_NYC,zorder=0, extent=BB);



# In[145]:


EHouston=speedlimit[(speedlimit['street']=='EAST HOUSTON STREET')]


# In[146]:


#speedlimit['street'].unique().tolist()


# In[147]:


#fig,axs=plt.subplots(1,figsize=(15,15))
#axs=EHouston['geometry'].plot(zorder=1, alpha=1,ax=axs,color='black')
#axs.set_ylim(40.5,40.9)
#axs.set_xlim(-74.25,-73.75)
#axs.imshow(map_NYC,zorder=0, extent=BB);


# In[148]:


collision_rate=pd.DataFrame(collision_df.groupby('HOUR')[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED','NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED']].sum())
collision_rate=collision_rate.reset_index()


# In[149]:


collision_rate_count=pd.DataFrame(collision_df.groupby('HOUR')['DATE'].count())
collision_rate_count=collision_rate_count.reset_index()


# In[150]:


collision_rate=pd.merge(collision_rate,collision_rate_count)


# In[151]:


collision_rate=collision_rate.rename(columns={'DATE':'COUNTS'})


# In[152]:


collision_rate['injury_rate']=collision_rate['NUMBER OF PERSONS INJURED']/collision_rate['COUNTS']
collision_rate['kill_rate']=collision_rate['NUMBER OF PERSONS KILLED']/collision_rate['COUNTS']
collision_rate['ped_injury_rate']=collision_rate['NUMBER OF PEDESTRIANS INJURED']/collision_rate['COUNTS']
collision_rate['ped_kill_rate']=collision_rate['NUMBER OF PEDESTRIANS KILLED']/collision_rate['COUNTS']
collision_rate['cyc_injury_rate']=collision_rate['NUMBER OF CYCLIST INJURED']/collision_rate['COUNTS']
collision_rate['cyc_kill_rate']=collision_rate['NUMBER OF CYCLIST KILLED']/collision_rate['COUNTS']


# In[153]:


inj_ensemble=['injury_rate','ped_injury_rate','cyc_injury_rate']


# In[154]:


#fig=plt.figure(figsize=(15,10))
#ax = fig.add_subplot(111)
#ax.bar(collision_rate['HOUR']-1/3,collision_rate['injury_rate'],width=1/3,label="Overall injury rate")
#ax.bar(collision_rate['HOUR'],collision_rate['ped_injury_rate'],width=1/3,label="Pedestrain injury rate")
#ax.bar(collision_rate['HOUR']+1/3,collision_rate['cyc_injury_rate'],width=1/3, label="Cyclist injury rate")
#ax.legend()
#ax.set_title('injury and hours',fontsize=25)
#plt.savefig("injury and hours.png")



# In[155]:


#fig=plt.figure(figsize=(10,6))
#ax = fig.add_subplot(111)
#ax.bar(collision_rate['HOUR']-1/3,collision_rate['kill_rate'],width=1/3,label="Overall death rate")
#ax.bar(collision_rate['HOUR'],collision_rate['ped_kill_rate'],width=1/3,label="Pedestrain death rate")
#ax.bar(collision_rate['HOUR']+1/3,collision_rate['cyc_kill_rate'],width=1/3, label="Cyclist death rate")
#ax.legend(fontsize=15)
#ax.set_title('death and hours',fontsize=25)
#plt.savefig("death and hours.png")



# In[156]:


from shapely.geometry import Point
from shapely.geometry import shape


# In[157]:


EHouston['geometry'][0:100].plot()


# In[158]:


type(EHouston['geometry'][0:1])


# In[159]:


from geopandas import GeoDataFrame
from shapely.geometry import Point


# In[160]:


geometry=[Point(xy) for xy in zip(collision_df['LONGITUDE'],collision_df['LATITUDE'])]
crs = {'init':'epsg:4326'}
gdf_CLS=GeoDataFrame(collision_df, crs=crs,geometry=geometry)


# In[161]:


EHouston['geometry'][0:100].plot()


# In[162]:


EHOUSTON_ACC=collision_df[collision_df['ON STREET NAME']=='EAST HOUSTON STREET             ']


# In[163]:


geometry=[Point(xy) for xy in zip(EHOUSTON_ACC['LONGITUDE'],EHOUSTON_ACC['LATITUDE'])]
crs = {'init':'epsg:4326'}
gdf_EH=GeoDataFrame(EHOUSTON_ACC, crs=crs,geometry=geometry)


# In[164]:


gdf_CLS['BOROUGH'].unique()


# In[165]:


gdf_null=gdf_CLS[gdf_CLS['BOROUGH'].isnull()]


# In[166]:


#gdf_null['geometry'].plot(figsize=(15,15))


# In[167]:


gdf_mht=gdf_CLS[gdf_CLS['BOROUGH']=='MANHATTAN']


# In[168]:



EHouston[EHouston['geometry'][0:100].distance(gdf_EH['geometry'][1355629])<0.00008].geometry


# In[169]:


gdf_EH_candid=gdf_CLS[(gdf_CLS['LATITUDE']<40.726)&(gdf_CLS['LATITUDE']>40.717)&(gdf_CLS['LONGITUDE']>-73.997)&(gdf_CLS['LONGITUDE']<-73.974)]


# In[170]:


#gdf_EH_candid['geometry'].plot()


# In[171]:


#len(EHouston[EHouston['geometry'][0:100].distance(EH_df.iloc[0]['geometry'])<0.00009])


# In[172]:


#Create a chunk of East Houston Street 
EHouston['buffer']=EHouston.buffer(0.00009)
from shapely.ops import cascaded_union
EHouston_merged=cascaded_union(EHouston['buffer'])


# In[175]:


#Doing it in this way, there are a lot of repetitive data 
#EH_df=pd.DataFrame()
#for item in gdf_EH_candid['geometry']:
#    if EHouston_merged.contains(item):
#        EH_df=EH_df.append(gdf_EH_candid[gdf_EH_candid['geometry']==item])


# In[176]:


gdf_EH=gdf_EH.reset_index()


# In[ ]:


#type(gdf_EH['geometry'][1])


# In[ ]:


# Try doing this differently, see if it works
#EH_df2=pd.DataFrame()
#for index,row in gdf_EH.iterrows():
#    if EHouston_merged.contains(row[31]):
#        EH_df2=EH_df2.append(gdf_EH.iloc[[index]])


# In[177]:


gdf_EH_candid=gdf_EH_candid.reset_index()


# In[178]:


#All accidents on Houston Street
EH_DF=pd.DataFrame()
for index,row in gdf_EH_candid.iterrows():
    if EHouston_merged.contains(row[31]):
        EH_DF=EH_DF.append(gdf_EH_candid.iloc[[index]])


# In[179]:


EH_DF.to_csv('East_Houston_accidents.csv')


# In[13]:


import mplleaflet


# In[181]:


EH_DF_injury=EH_DF[EH_DF['NUMBER OF PERSONS INJURED']>0]


# In[182]:


EH_DF_kill=EH_DF[EH_DF['NUMBER OF PERSONS KILLED']>0]


# In[183]:


#All accidents on google map
fig, axs=plt.subplots(1)
axs=EH_DF[(EH_DF['NUMBER OF PERSONS INJURED']==0)&(EH_DF['NUMBER OF PERSONS KILLED']==0)]['geometry'].plot(markersize=15, alpha=0.3, color='red',marker=11,ax=axs)
axs=EH_DF_injury[EH_DF_injury['NUMBER OF PEDESTRIANS INJURED']>0]['geometry'].plot(markersize=20, alpha=0.3, color='green',marker='X',ax=axs)
axs=EH_DF_injury[EH_DF_injury['NUMBER OF CYCLIST INJURED']>0]['geometry'].plot(markersize=20, alpha=0.3, color='cyan',marker='*',ax=axs)
axs=EH_DF_injury[EH_DF_injury['NUMBER OF MOTORIST INJURED']>0]['geometry'].plot(markersize=20, alpha=0.3, color='blue',marker='H',ax=axs)
axs=EH_DF_kill['geometry'].plot(markersize=25, alpha=0.9, color='black',marker='X',ax=axs)
mplleaflet.show()


# In[184]:


EH_DF['geometry']


# In[185]:


EH_DF_injury[['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED']].sum()


# In[186]:


EH_DF_count=pd.DataFrame(EH_DF.groupby('HOUR')[['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED']].sum())


# In[187]:


EH_DF_count=EH_DF_count.reset_index()


# In[188]:


#fig=plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#ax.bar(EH_DF_count['HOUR']-1/3,EH_DF_count['NUMBER OF PEDESTRIANS INJURED'], width=1/3,label="NUMBER OF PEDESTRIANS INJURED")
#ax.bar(EH_DF_count['HOUR'],EH_DF_count['NUMBER OF CYCLIST INJURED'],width=1/3,label="NUMBER OF CYCLIST INJURED")
#ax.bar(EH_DF_count['HOUR']+1/3,EH_DF_count['NUMBER OF MOTORIST INJURED'], width=1/3, label="NUMBER OF MOTORIST INJURED")
#ax.legend(fontsize=10)
#ax.set_title('East Houston Street hours and casualty',fontsize=25)
#plt.savefig("East Houston Street hours and casualty.png")




# In[189]:


import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
map_EHS = Image.open("/Users/yuantaoli/Downloads/traffic data/E H st.png")
BB = (-73.997564, -73.974400, 40.717870, 40.725870)
mpl.rcParams['figure.dpi']= 300
plt.figure(figsize=(17,6.5))
plt.imshow(map_EHS,zorder=0, extent=BB);


# In[195]:


fig,axs = plt.subplots(24,figsize=(17,156))
for x in range(24):
    axs[x]=EH_DF[(EH_DF['HOUR']==x)&(EH_DF['NUMBER OF PERSONS INJURED']==0)&(EH_DF['NUMBER OF PERSONS KILLED']==0)]['geometry'].plot(markersize=200, alpha=0.3, color='red',marker=11,ax=axs[x],zorder=1,label='collisions with no injury/death')
    axs[x]=EH_DF_injury[(EH_DF_injury['HOUR']==x)&(EH_DF_injury['NUMBER OF PEDESTRIANS INJURED']>0)]['geometry'].plot(markersize=200, alpha=0.2, color='green',marker='X',ax=axs[x],zorder=1,label='pedestrian injury')
    if (x!=3)&(x!=5)&(x!=8):
        axs[x]=EH_DF_injury[(EH_DF_injury['HOUR']==x)&(EH_DF_injury['NUMBER OF CYCLIST INJURED']>0)]['geometry'].plot(markersize=200, alpha=0.2, color='cyan',marker='*',ax=axs[x],zorder=1,label='cyclist injury')
    axs[x]=EH_DF_injury[(EH_DF_injury['HOUR']==x)&(EH_DF_injury['NUMBER OF MOTORIST INJURED']>0)]['geometry'].plot(markersize=200, alpha=0.2, color='blue',marker='H',ax=axs[x],zorder=1,label='motorist injury')
    if (x==9)|(x==16)|(x==21):
        axs[x]=EH_DF_kill[EH_DF_kill['HOUR']==x]['geometry'].plot(markersize=300, alpha=0.9, color='black',marker='X',ax=axs[x],zorder=1,label='death')
    axs[x].imshow(map_EHS,zorder=0, extent=BB);
    axs[x].set_ylim(40.717870,40.725870)
    axs[x].set_xlim(-73.997564, -73.974400)
    axs[x].legend(loc='upper right',prop={'size': 15})
    axs[x].set_title('Accidents at {}'.format(x))


    
        #mplleaflet.show()


# In[196]:


fig,axs = plt.subplots(1,figsize=(17,6.5))
for x in range(24):
    axs=EH_DF[(EH_DF['HOUR']==x)&(EH_DF['NUMBER OF PERSONS INJURED']==0)&(EH_DF['NUMBER OF PERSONS KILLED']==0)]['geometry'].plot(markersize=200, alpha=0.3, color='red',marker=11,ax=axs,zorder=1,label='collisions with no injury/death')
    axs=EH_DF_injury[(EH_DF_injury['HOUR']==x)&(EH_DF_injury['NUMBER OF PEDESTRIANS INJURED']>0)]['geometry'].plot(markersize=200, alpha=0.2, color='green',marker='X',ax=axs,zorder=1,label='pedestrian injury')
    if (x!=3)&(x!=5)&(x!=8):
        axs=EH_DF_injury[(EH_DF_injury['HOUR']==x)&(EH_DF_injury['NUMBER OF CYCLIST INJURED']>0)]['geometry'].plot(markersize=200, alpha=0.2, color='cyan',marker='*',ax=axs,zorder=1,label='cyclist injury')
    axs=EH_DF_injury[(EH_DF_injury['HOUR']==x)&(EH_DF_injury['NUMBER OF MOTORIST INJURED']>0)]['geometry'].plot(markersize=200, alpha=0.2, color='blue',marker='H',ax=axs,zorder=1,label='motorist injury')
    if (x==9)|(x==16)|(x==21):
        axs=EH_DF_kill[EH_DF_kill['HOUR']==x]['geometry'].plot(markersize=300, alpha=0.9, color='black',marker='X',ax=axs,zorder=1,label='death')
    axs.imshow(map_EHS,zorder=0, extent=BB);
    axs.set_ylim(40.717870,40.725870)
    axs.set_xlim(-73.997564, -73.974400)
    axs.legend(loc='upper right',prop={'size': 15})
    axs.set_title('Accidents at {}'.format(x))
    fig = plt.gcf()
    fig.savefig('houston accidents at {}'.format(x))

    
        #mplleaflet.show()


# In[191]:


EH_DF_injury[['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED','CONTRIBUTING FACTOR VEHICLE 1']].groupby('CONTRIBUTING FACTOR VEHICLE 1').sum()


# In[192]:


EH_DF.groupby('CONTRIBUTING FACTOR VEHICLE 1').count()


# In[193]:


#EH_ASS=pd.DataFrame()
#for index,row in street_ass.iterrows():
#    if EHouston_merged.contains(row[31]):
#        EH_DF=EH_DF.append(gdf_EH_candid.iloc[[index]])


# In[194]:



Street_ass

