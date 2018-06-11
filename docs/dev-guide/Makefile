DOXYCONF=Doxygen_settings
LOGO=img/logo.png
export LOGO

html: $(DOXYCONF) $(LOGO)
	$(CMDECHO)doxygen $(DOXYCONF) > make_docs.log 2>&1

$(LOGO):
	$(CMDECHO)mkdir -p img && \
		wget http://www.gpusph.org/$@ -O $@

clean:
	rm -rf $(LOGO) html

# We consider html a phony target so it gets rebuilt unconditionally, otherwise
# we'd have to make it depend on all the source files
.PHONY: clean html

.NOTPARALLEL:
