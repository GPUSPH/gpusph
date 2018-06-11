# option: echo - 0 silent, 1 show commands
ifeq ($(echo), 1)
	CMDECHO :=
else
	CMDECHO := @
endif

DOXYCONF=Doxygen_settings
LOGO=img/logo.png

html: $(DOXYCONF) $(LOGO)
	$(CMDECHO)doxygen $(DOXYCONF) > make_docs.log 2>&1

img/logo.png:
	$(CMDECHO)mkdir -p img && \
		wget http://www.gpusph.org/img/logo.png -O img/logo.png

clean:
	rm -rf img/logo.png html

# We consider html a phony target so it gets rebuilt unconditionally, otherwise
# we'd have to make it depend on all the source files
.PHONY: clean html

.NOTPARALLEL:
