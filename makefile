ifndef OUTPUT
ifdef TMPDIR
OUTPUT:=$(TMPDIR)/a.out
else
OUTPUT:=/data/user/0/org.c.ide/files/tmpdir/a.out
endif
endif

Directories:=. caffe_mods util/lib_json util env ui # ui/pd
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

USECAFFE=1
ifdef USECAFFE
LDFLAGS+= -L/sdcard/_ws/deep/thrid/lib64 -lcaffeproto -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive -lopenblas -lprotobuf -lglog -lgflags
endif
BaseName:=$(basename $(Sources))
CPPFLAGS+= -DCPU_ONLY -DNDEBUG -DGOOGLE_STRIP_LOG=2 -Iutil -I. -I/sdcard/_ws/deep/thrid/bullet3 -I/sdcard/_ws/deep/thrid/include
ifdef BaseName
all.suffix:= $(suffix $(Sources))
ifeq ($(words $(BaseName)), $(words $(all.suffix)))
cxx.suffix:= $(filter .cc .cpp .cxx, $(all.suffix))
XCC:= $(if $(cxx.suffix), $(CXX),$(CC))
Objects:=$(BaseName:%=%.o)
all: $(OUTPUT)
#	@echo $(Objects) $(CPPFLAGS)
$(OUTPUT):a.out
	cp $^ $@
	chmod 700 $@
a.out:$(Objects)
	$(XCC) $^ $(LDFLAGS) -o $@ -landroid -llog -lbox2d -lEGL -lGLESv3
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