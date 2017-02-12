for i in {1..500}
do
	gnuplot -e "set terminal jpeg; splot 'data$i.ds' " > pic$i.jpeg 
done

ffmpeg -i pic%04d.jpeg movie.mpeg