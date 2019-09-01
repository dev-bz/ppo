ifndef OUTPUT
OUTPUT:=/data/user/0/org.c.ide/files/tmpdir/a.out
endif
Directories:=. ../util/lib_json ../util pd
FindSources:=$(Directories:%=%/*.c) $(Directories:%=%/*.cc) $(Directories:%=%/*.cxx) $(Directories:%=%/*.cpp)
RMObjects:= $(wildcard $(Directories:%=%/*.o))
Sources:=$(wildcard $(FindSources)) $(MORE)
ifdef MORE
	MoreBaseName:=$(sort $(basename $(MORE)))
endif
MoreObjects:=$(wildcard $(MoreBaseName:%=%.o) $(OUTPUT))
ifdef MoreObjects
RMObjects+=$(MoreObjects)
endif
ifdef RMObjects
RMObjects:=@rm $(RMObjects)
endif

BaseName:=$(basename $(Sources))
CPPFLAGS+= -I. -I..
ifdef BaseName
all.suffix:= $(suffix $(Sources))
ifeq ($(words $(BaseName)), $(words $(all.suffix)))
cxx.suffix:= $(filter .cc .cpp .cxx, $(all.suffix))
XCC:= $(if $(cxx.suffix), $(CXX),$(CC))
Objects:=$(BaseName:%=%.o)

all: $(OUTPUT)

$(OUTPUT):a.out
	cp $^ $@
	chmod 700 $@
a.out:$(Objects)
	$(XCC) $^ $(LDFLAGS) -o $@ -landroid -lbox2d -lEGL -lGLESv3 -L/sdcard/_ws/jsoncpp/thrid/lib64
	@echo $(XCC) $^ $(LDFLAGS) -o $@ -lEGL -lGLESv3
else
all:
	clear
	@echo 某些文件名含有空格,终止编译
endif
else
all:
	clear
	@echo 项目中没有找到源代码文件
endif
clean:
	$(RMObjects)
%.o: %.c %.cc %.cxx %.cpp
	@echo $<