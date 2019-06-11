# ggplot is popular plotting library, ggmap is for google maps
library(ggplot2)
library(ggmap)
library(spatstat)

# read in robbery data from Indy, 
# you can download this data (and for other crimes) from Socrata
rob=read.csv("~/Downloads/robbery.csv")

# create a histogram
minlon=min(rob$longitude)
minlat=min(rob$latitude)
maxlon=max(rob$longitude)
maxlat=max(rob$latitude)

X <- ppp(rob$longitude,rob$latitude,window=owin(c(minlon,maxlon),c(minlat,maxlat)))
plot(X)
plot(quadratcount(X, nx = 10, ny = 10),add = TRUE, cex = 2)

# put points on a google map
# only add a map if you get google api key 
center_latlon <- c(mean(rob$longitude),mean(rob$latitude))

key <- 'AIzaSyDFHV9k2AhfXgwjQd5633jynu0NnJenvhA'
register_google(key = key)

Map<-get_map(center_latlon, zoom = 13)
DowntownMap <- ggmap(Map)

# without api key just replace "DowntownMap" with ggplot()

# scatter plot
DowntownMap+geom_point(data=rob,aes(x=longitude,y=latitude))

# kde heat map
DowntownMap+stat_density2d(aes(x = longitude, y = latitude, fill = ..level..,
                      alpha=..level..),
                      bins = 20, data = rob,h=.01,
                      geom = "polygon")+
                      scale_fill_gradientn(colours=c("yellow","orange","red"))



# append census data to crime data
library(tigris)
library(leaflet)

indy <- block_groups(state = "IN", county = "Marion")

library(totalcensus)

# replace with path to the census data and codes
set_path_to_census("~/Desktop/desktop_rt/my_census_data")
codes=read.csv("~/Desktop/desktop_rt/codes.csv")

acsdata <- read_acs5year(
  year = 2015,
  states = "IN",
  table_contents=as.character(codes$codes),
  summary_level = "block group"
) 


geoids=cSplit(as.data.table(acsdata), "GEOID", "US")


geoids=geoids[is.element(geoids$GEOID_2,indy$GEOID),]
geoids$GEOID=geoids$GEOID_2

rob$x=rob$longitude
rob$y=rob$latitude


coordinates(rob)<-~x+y
proj4string(rob) <- CRS("+proj=longlat +ellps=WGS84") 
poly.proj <- proj4string(indy) 
rob <- spTransform(rob,CRS(poly.proj)) 

temp=over(rob,indy)
rob=as.data.frame(rob)
temp=as.data.frame(temp)
rob=cbind(rob,temp)

rob=merge(rob,geoids,by="GEOID")

rob$count = 1
rob_agg = aggregate(count~GEOID, data=rob, FUN=sum)

rob_reg_data=merge(rob, rob_agg, by="GEOID", all.x=T)

model=glm(count.y~B27010_064+C17002_003, data=rob_reg_data, family = "poisson")
