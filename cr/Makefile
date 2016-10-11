CC = g++ -Wall
#CC = g++ -ansi -Wall -pedantic
#CFLAGS = -g -Wall -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
#CFLAGS = -g -Wall
LDFLAGS = -lgsl -lm -lgslcblas

GSL_INCLUDE = ~/.local/include
GSL_LIB = ~/.local/lib

LSOURCE = main.cpp utils.cpp cr.cpp data.cpp
LHEADER = utils.h cr.h data.h

cr: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o $@ $(LDFLAGS)

cr-d: $(LSOURCE) $(HEADER)
	  $(CC) -g -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	-rm -f *.o cr
clean-d:
	-rm -f *.o cr-d
