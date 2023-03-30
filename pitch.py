########################################################################################################################
# SOURCE OF DATA
"""
Gyódi, Kristóf, & Nawaro, Łukasz. (2021).
Determinants of Airbnb prices in European cities:
A spatial econometrics approach (Supplementary Material)
[Data set]. Zenodo. https://doi.org/10.5281/zenodo.4446043

https://www.sciencedirect.com/science/article/pii/S0261517721000388?via%3Dihub
"""

"""
chosen dataset:
    either one city or all cities
    hypothesis: price is more dependant on spatial features than quality of an Airbnb
    goal: price as target for regression prediction
    >>> features:
        dist, metro_dist, rest_index, attr_index, (city?), [spatial features]
        cleanliness, satisfaction, weekend, [quality features]
        host_is_superhost, room_type, person_capacity [basic features]
    method:
        data pre-processing as a whole group
        split up into two teams: 
        one group predicts based on spatial and base features, the other group based on quality and base features
        compare: which types of features are stronger indicators for the pricing?
    project schedule: 
        duration: 2.5 months (10 weeks) (beginning of April - mid june)
        2 - 3 weeks pre-processing
        6 weeks predict - compare - iterate
        1 week prepare final presentation
    pitch:
        present datasets (regression namedropping) (Nico)
        explain why features and targets were chosen (Jakob)
        explain our hypothesis and method of prediction based on two types of feature categories (Marlis)
        tentative schedule (Lorenz)
    request feedback:
        either one city or all cities?
        all room_types or just the two most common?
        how to throw longitude/latitude into the mix
        
    feedback:
        splitting of features makes sense -> yay
        try different combinations: each of the three categories solo, each combination, drop single features
        compare and check effect on accuracy
        focus on Vienna
        longitude/latitude works better if we concentrate on 1 city -> yay
        important finding: find out which feature(s) are relevant
        do:
            df.describe and other useful functions of pandas dataframes
            boxplot of each numerical feature
"""

########################################################################################################################
# IMPORTS
import pandas as pd
from math import pi

from bokeh.models import LinearAxis, ColorBar, ColorMapper, LinearColorMapper, TileSource
from bokeh.palettes import Category20c, mpl
from bokeh.plotting import figure, show
from bokeh.transform import cumsum, linear_cmap
from bokeh.layouts import row, layout
from bokeh.io import output_file
import math
from bokeh.models.tiles import WMTSTileSource

########################################################################################################################
# ARRANGE DFS
df1 = pd.read_csv('datasets/amsterdam_weekdays.csv', sep=',')
df1['weekend'] = False
df2 = pd.read_csv('datasets/amsterdam_weekends.csv', sep=',')
df2['weekend'] = True
df1['city'] = 'Amsterdam'
df2['city'] = 'Amsterdam'
amsterdam = pd.concat([df1, df2], ignore_index=True, sort=False)
weekdays = df1
weekends = df2

df3 = pd.read_csv('datasets/athens_weekdays.csv', sep=',')
df3['weekend'] = False
df4 = pd.read_csv('datasets/athens_weekends.csv', sep=',')
df4['weekend'] = True
df3['city'] = 'Athens'
df4['city'] = 'Athens'
athens = pd.concat([df3, df4], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df3], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df4], ignore_index=True, sort=False)

df5 = pd.read_csv('datasets/barcelona_weekdays.csv', sep=',')
df5['weekend'] = False
df6 = pd.read_csv('datasets/barcelona_weekends.csv', sep=',')
df6['weekend'] = True
df5['city'] = 'Barcelona'
df6['city'] = 'Barcelona'
barcelona = pd.concat([df5, df6], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df5], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df6], ignore_index=True, sort=False)

df7 = pd.read_csv('datasets/berlin_weekdays.csv', sep=',')
df7['weekend'] = False
df8 = pd.read_csv('datasets/berlin_weekends.csv', sep=',')
df8['weekend'] = True
df7['city'] = 'Berlin'
df8['city'] = 'Berlin'
berlin = pd.concat([df7, df8], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df7], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df8], ignore_index=True, sort=False)

df9 = pd.read_csv('datasets/budapest_weekdays.csv', sep=',')
df9['weekend'] = False
df10 = pd.read_csv('datasets/budapest_weekends.csv', sep=',')
df10['weekend'] = True
df9['city'] = 'Budapest'
df10['city'] = 'Budapest'
budapest = pd.concat([df9, df10], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df9], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df10], ignore_index=True, sort=False)

df11 = pd.read_csv('datasets/lisbon_weekdays.csv', sep=',')
df11['weekend'] = False
df12 = pd.read_csv('datasets/lisbon_weekends.csv', sep=',')
df12['weekend'] = True
df11['city'] = 'Lisbon'
df12['city'] = 'Lisbon'
lisbon = pd.concat([df11, df12], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df11], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df12], ignore_index=True, sort=False)

df13 = pd.read_csv('datasets/london_weekdays.csv', sep=',')
df13['weekend'] = False
df14 = pd.read_csv('datasets/london_weekends.csv', sep=',')
df14['weekend'] = True
df13['city'] = 'London'
df14['city'] = 'London'
london = pd.concat([df13, df14], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df13], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df14], ignore_index=True, sort=False)

df15 = pd.read_csv('datasets/paris_weekdays.csv', sep=',')
df15['weekend'] = False
df16 = pd.read_csv('datasets/paris_weekends.csv', sep=',')
df16['weekend'] = True
df15['city'] = 'Paris'
df16['city'] = 'Paris'
paris = pd.concat([df15, df16], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df15], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df16], ignore_index=True, sort=False)

df17 = pd.read_csv('datasets/rome_weekdays.csv', sep=',')
df17['weekend'] = False
df18 = pd.read_csv('datasets/rome_weekends.csv', sep=',')
df18['weekend'] = True
df17['city'] = 'Rome'
df18['city'] = 'Rome'
rome = pd.concat([df17, df18], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df17], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df18], ignore_index=True, sort=False)

df19 = pd.read_csv('datasets/vienna_weekdays.csv', sep=',')
df19['weekend'] = False
df20 = pd.read_csv('datasets/vienna_weekends.csv', sep=',')
df20['weekend'] = True
df19['city'] = 'Vienna'
df20['city'] = 'Vienna'
vienna = pd.concat([df19, df20], ignore_index=True, sort=False)
weekdays = pd.concat([weekdays, df19], ignore_index=True, sort=False)
weekends = pd.concat([weekends, df20], ignore_index=True, sort=False)

all_cities = pd.concat([amsterdam, athens, barcelona, berlin, budapest, lisbon, london, paris, rome, vienna],
                       ignore_index=True, sort=False)
del df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20

########################################################################################################################
# PIE CHART
x = {
    'Amsterdam': len(amsterdam),
    'Athens': len(athens),
    'Barcelona': len(barcelona),
    'Berlin': len(berlin),
    'Budapest': len(budapest),
    'Lisbon': len(lisbon),
    'London': len(london),
    'Paris': len(paris),
    'Rome': len(rome),
    'Vienna': len(vienna),
}

data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
data['angle'] = data['value'] / data['value'].sum() * 2 * pi
data['color'] = Category20c[len(x)]

plot1 = figure(height=400, title='airbnbs per city', toolbar_location=None,
               tools='hover', tooltips='@country: @value', x_range=(-0.5, 1.0))

plot1.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color='white', fill_color='color', legend_field='country', source=data)

plot1.axis.axis_label = None
plot1.axis.visible = False
plot1.grid.grid_line_color = None

plot1.margin = 10
########################################################################################################################
# SCATTER PLOT CLEANLINESS - SATISFACTION
plot2 = figure(width=400, height=400)
# add a circle renderer with a size, color, and alpha
plot2.circle(all_cities['cleanliness_rating'], all_cities['guest_satisfaction_overall'], size=2, color='navy', alpha=0.5)
plot2.xaxis.axis_label = 'cleanliness'
plot2.yaxis.axis_label = 'guest satisfaction'
plot2.margin = 10
########################################################################################################################
# SCATTER PLOT CITY - PRICE MEAN & MODE
cities = ['Amsterdam', 'Athens', 'Barcelona',
          'Berlin', 'Budapest', 'Lisbon', 'London',
          'Paris', 'Rome', 'Vienna']
prices_mean = [amsterdam.loc[:, 'realSum'].mean(),
               athens.loc[:, 'realSum'].mean(),
               barcelona.loc[:, 'realSum'].mean(),
               berlin.loc[:, 'realSum'].mean(),
               budapest.loc[:, 'realSum'].mean(),
               lisbon.loc[:, 'realSum'].mean(),
               london.loc[:, 'realSum'].mean(),
               paris.loc[:, 'realSum'].mean(),
               rome.loc[:, 'realSum'].mean(),
               vienna.loc[:, 'realSum'].mean()]
prices_mode = [amsterdam.loc[:, 'realSum'].mode()[0],
               athens.loc[:, 'realSum'].mode()[0],
               barcelona.loc[:, 'realSum'].mode()[0],
               berlin.loc[:, 'realSum'].mode()[0],
               budapest.loc[:, 'realSum'].mode()[0],
               lisbon.loc[:, 'realSum'].mode()[0],
               london.loc[:, 'realSum'].mode()[0],
               paris.loc[:, 'realSum'].mode()[0],
               rome.loc[:, 'realSum'].mode()[0],
               vienna.loc[:, 'realSum'].mode()[0]]

plot3 = figure(x_range=cities, height=400)
# add a circle renderer with a size, color, and alpha
plot3.circle(pd.Series(cities), pd.Series(prices_mean), size=20, color='navy', alpha=0.5, legend_label='mean')
plot3.circle(pd.Series(cities), pd.Series(prices_mode), size=20, color='red', alpha=0.5, legend_label='mode')
plot3.xaxis.axis_label = 'city'
plot3.yaxis.axis_label = 'price (EUR)'
plot3.legend.location = 'top_right'
plot3.legend.border_line_width = 3
plot3.legend.border_line_color = 'black'
plot3.margin = 10
########################################################################################################################
# SCATTER PLOT CITY - PRICES
plot4 = figure(width=800, height=800)
plot4.circle(all_cities['dist'], all_cities['realSum'],
             size=2, color='navy', alpha=0.5, legend_label='distance to city center')
plot4.circle(all_cities['metro_dist'], all_cities['realSum'],
             size=2, color='red', alpha=0.5, legend_label='distance to metro')
plot4.yaxis.axis_label = 'price (EUR)'
plot4.xaxis.axis_label = 'distance (km)'
plot4.legend.location = 'top_right'
plot4.legend.border_line_width = 3
plot4.legend.border_line_color = 'black'
plot4.margin = 10

########################################################################################################################
# PIE CHART CLEANLINESS - DISTANCE
"""
dsmall = all_cities.where(all_cities['dist'] < 3)
clgood = all_cities.where(all_cities['cleanliness_rating'] >= 9)
dsmallprice = dsmall[dsmall['realSum'] >= 2000]
clgoodprice = clgood[clgood['realSum'] >= 2000]

dat = {
    'distance to city center < 3 km': len(dsmallprice),
    'cleanliness rating >= 9 stars': len(clgoodprice)
}

data = pd.Series(dat).reset_index(name='value').rename(columns={'index': 'feature'})
data['angle'] = data['value'] / data['value'].sum() * 2 * pi
data['color'] = 'green', 'orange'

plot5 = figure(height=400, title='Number of high priced Airbnbs (>1000 EUR/night)', toolbar_location=None, # > 2000 EUR for 2 nights
               tools='hover', tooltips='@feature: @value', x_range=(-0.5, 1.0))

plot5.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color='white', fill_color='color', legend_field='feature', source=data)

plot5.axis.axis_label = None
plot5.axis.visible = False
plot5.grid.grid_line_color = None

plot5.margin = 10
"""
# SCATTER PLOTS CLEANLINESS/DISTANCE - PRICES
plot5 = figure(width=800, height=800)
plot6 = figure(width=800, height=800)


plot5.circle(all_cities['dist'], all_cities['realSum'],
             size=2, color='purple', alpha=0.5, legend_label='distance to city center')

plot6.circle(all_cities['cleanliness_rating'], all_cities['realSum'],
             size=2, color='green', alpha=0.5, legend_label='cleanliness rating')

plot5.yaxis.axis_label = 'price (EUR)'
plot5.xaxis.axis_label = 'distance (km)'
plot5.legend.location = 'top_right'
plot5.legend.border_line_width = 3
plot5.legend.border_line_color = 'black'
plot5.margin = 10

plot6.yaxis.axis_label = 'price (EUR)'
plot6.xaxis.axis_label = 'rating (stars)'
plot6.legend.location = 'top_right'
plot6.legend.border_line_width = 3
plot6.legend.border_line_color = 'black'
plot6.margin = 10

########################################################################################################################
# MAP WITH AIRBNB LOCATIONS IN VIENNA + PRICE COLOR SCALE

# helper functions to convert lat/long to mercator coordinates
def mercator_y(a):
    return math.log(math.tan(math.pi / 4 + math.radians(a) / 2)) * 6378137.0

def mercator_x(a):
    return math.radians(a) * 6378137.0


# configure data source for map
tile_source = WMTSTileSource(
    url='https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{@2x}.png'
)

plot7 = figure(x_range=(1812587, 1832587), y_range=(6131582, 6151582),
               x_axis_type="mercator", y_axis_type="mercator", height=600, width=700)
plot7.add_tile(tile_source)

color_mapper123 = LinearColorMapper(palette=['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'], low=70, high=400)
vienna['mercator_x'] = vienna['lng'].apply(lambda lng: mercator_x(lng))
vienna['mercator_y'] = vienna['lat'].apply(lambda lat: mercator_y(lat))
plot7.circle(x='mercator_x', y='mercator_y', source=vienna, size=5, fill_color={'field': 'realSum', 'transform': color_mapper123}, alpha=0.5, line_color=None)
color_bar = ColorBar(color_mapper=color_mapper123, location=(0,0))
plot7.add_layout(color_bar, 'right')
plot7.xaxis.axis_label = 'longitude'
plot7.yaxis.axis_label = 'latitude'
plot7.margin = 10

# show the results
output_file('charts.html')
show(layout([plot1, plot3, plot2], [plot4, plot5, plot6], [plot7]))

# i am writing a long comment and wanna see to which repository this is being pushed