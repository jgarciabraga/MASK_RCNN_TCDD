rm(list=ls())
library(sp)
library(rgdal)
library(rgeos)
library(raster)

#h_lines = rgdal::readOGR('C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Shapes/double_horizontal_lines.shp')
#v_lines = rgdal::readOGR('C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Shapes/double_vertical_lines.shp')
#trees = rgdal::readOGR('C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Delineação/resposta_t2_158_red_uni.shp')
#h_lines = rgdal::readOGR('C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Shapes/square_hori_lines.shp')
#v_lines = rgdal::readOGR('C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Shapes/square_vert_lines.shp')
#trees = rgdal::readOGR('C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Delineação/temp/temp.shp')

h_lines = rgdal::readOGR('C:/Users/Projeto/OneDrive - inpe.br/ITCD/Shapes/h_lines.shp')
v_lines = rgdal::readOGR('C:/Users/Projeto/OneDrive - inpe.br/ITCD/Shapes/v_lines.shp')
trees = rgdal::readOGR('C:/Users/Projeto/Desktop/GUIT/dataset/trees/test_result/response.shp')
df = data.frame(id=1:length(trees))
trees@data = df

mat_inter_tress_hor = rgeos::gIntersects(h_lines,trees,byid = TRUE)
mat_inter_tress_hor = mat_inter_tress_hor[,1]

trees_h = trees[mat_inter_tress_hor, ]
x_coord = data.frame(x=rep(NA, length(trees_h)))
y_coord = data.frame(y=rep(NA, length(trees_h)))

for(cont in 1:length(trees_h)){
  pol = trees_h[cont, ]
  cent_pol = rgeos::gCentroid(pol)
  mat = extent(cent_pol)
  x_coord$x[cont] = mat@xmin
  y_coord$y[cont] = mat@ymin
}

trees_h$cent_x_coord = x_coord$x
trees_h$cent_y_coord = y_coord$y
check = TRUE

for(cont in 1:length(trees_h)){
  pol = trees_h[cont, ]
  aux = trees_h[which(trees_h$id != pol$id), ]
  x_pol = pol$cent_x_coord
  y_pol = pol$cent_y_coord
  dist_fim = 1000
  id_fim = 0
  for(k in 1:length(aux)){
    temp = aux[k, ]
    x_temp = temp$cent_x_coord
    y_temp = temp$cent_y_coord
    dist = base::sqrt((x_temp-x_pol)^2 + (y_temp-y_pol)^2)
    if(dist < dist_fim){
      dist_fim = dist
      id_fim = temp$id
    }
  }
  rm(temp)
  rm(aux)
  temp = trees_h[which(trees_h$id==id_fim), ]
  x_temp = temp$cent_x_coord
  dist = base::sqrt((x_temp-x_pol)^2)
  if(dist <= 2){
    new = rgeos::gUnion(pol,temp, byid = T)
    pid <- sapply(slot(new, "polygons"), function(x) slot(x, "ID"))
    df = data.frame(id=1:length(pid), row.names = pid)
    new = sp::SpatialPolygonsDataFrame(new, df)
    if(check){
      resp = new
      check = FALSE
    }
    else{
      resp = rbind(resp,new)
    }
    trees = trees[which(trees$id != temp$id), ]
    trees = trees[which(trees$id != pol$id), ]
  }
  print(cont)
}

mat_inter_tress_ver = rgeos::gIntersects(v_lines, trees, byid = TRUE)
mat_inter_tress_ver = mat_inter_tress_ver[,1]

trees_v = trees[mat_inter_tress_ver, ]
x_coord = data.frame(x=rep(NA, length(trees_v)))
y_coord = data.frame(y=rep(NA, length(trees_v)))

for(cont in 1:length(trees_v)){
  pol = trees_v[cont, ]
  cent_pol = rgeos::gCentroid(pol)
  mat = extent(cent_pol)
  x_coord$x[cont] = mat@xmin
  y_coord$y[cont] = mat@ymin
}

trees_v$cent_x_coord = x_coord$x
trees_v$cent_y_coord = y_coord$y
check = TRUE

for(cont in 1:length(trees_v)){
  pol = trees_v[cont, ]
  aux = trees_v[which(trees_v$id != pol$id), ]
  x_pol = pol$cent_x_coord
  y_pol = pol$cent_y_coord
  dist_fim = 1000
  id_fim = 0
  for(k in 1:length(aux)){
    temp = aux[k, ]
    x_temp = temp$cent_x_coord
    y_temp = temp$cent_y_coord
    dist = base::sqrt((x_temp-x_pol)^2 + (y_temp-y_pol)^2)
    if(dist < dist_fim){
      dist_fim = dist
      id_fim = temp$id
    }
  }
  rm(temp)
  rm(aux)
  temp = trees_v[which(trees_v$id==id_fim), ]
  y_temp = temp$cent_y_coord
  dist = base::sqrt((y_temp-y_pol)^2)
  if(dist <= 2){
    new = rgeos::gUnion(pol,temp, byid = T)
    pid <- sapply(slot(new, "polygons"), function(x) slot(x, "ID"))
    df = data.frame(id=1:length(pid), row.names = pid)
    new = sp::SpatialPolygonsDataFrame(new, df)
    if(check){
      resp_2 = new
      check = FALSE
    }
    else{
      resp_2 = rbind(resp_2,new)
    }
    trees = trees[which(trees$id != temp$id), ]
    trees = trees[which(trees$id != pol$id), ]
  }
  print(cont)
}

resp_final = rbind(resp, resp_2)

id_resp_final = 1:length(resp_final)
remover = rep(0,length(resp_final))
df_resp_final = data.frame(id_resp_final,remover)
resp_final@data = df_resp_final
removidos = 0

for(cont in 1:length(resp_final)){
  pol = resp_final[cont, ]
  if(pol$remover == 0){
    others = resp_final[which(resp_final$id_resp_final!=pol$id_resp_final), ]
    for(k in 1:length(others)){
      if(rgeos::gContains(pol, others[k, ])){
        resp_final@data[resp_final$id_resp_final==others[k, ]$id_resp_final, 'remover'] = 1
        removidos = removidos + 1
        print(removidos)
      }
    }  
  }
  
}

resp_final_final = resp_final[which(resp_final$remover == 0), ]
id = 1:length(resp_final_final)
df = data.frame(id)
resp_final_final@data = df


trees = rbind(trees,resp_final_final)
df = data.frame(id=1:length(trees))
trees@data = df

#rgdal::writeOGR(resp, 'C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Delineação/temp_regiao/tress_lin_squ_h.shp',layer = 'trees_h', driver = "ESRI Shapefile", overwrite_layer = TRUE, check_exists = TRUE)
#rgdal::writeOGR(resp_2, 'C:/Users/USUARIO/OneDrive - inpe.br/ITCD/Delineação/temp_regiao/tress_lin_squ_ver.shp',layer = 'trees_v', driver = "ESRI Shapefile", overwrite_layer = TRUE, check_exists = TRUE)
rgdal::writeOGR(trees, 'C:/Users/Projeto/Desktop/GUIT/dataset/trees/test_result/response_final.shp',layer = 'trees', driver = "ESRI Shapefile", overwrite_layer = TRUE, check_exists = TRUE)
