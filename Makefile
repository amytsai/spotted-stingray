CC = g++
ifeq ($(shell sw_vers 2>/dev/null | grep Mac | awk '{ print $$2}'),Mac)
	LDFLAGS = -L ./FreeImage/ -lfreeimage \
			 -I ./FreeImage/
    CFLAGS = -I ./FreeImage/
else
	LDFLAGS = -L ./FreeImage -lfreeimage \
			-I ./FreeImage
endif
	
RM = /bin/rm -f 
all: main 
main: raytrace.o
	$(CC) $(CFLAGS) -o raytrace raytrace.o $(LDFLAGS)
raytrace.o: raytrace.cpp
	$(CC) $(CFLAGS) -c raytrace.cpp -o raytrace.o
clean: 
	$(RM) *.o as1
 

