PROJ_PATH=workspace/projects/detectron2

define clean_repo
	git rm -r --cached .
	find '.' -name '*pyc' -exec rm -r {} +
	find '.' -name '*DS_Store' -exec rm -r {} +
	find '.' -name '*__pycache__' -exec rm -r {} +
endef

define sync_repo_to
	$(call clean_repo)
	rsync -av ~/$(PROJ_PATH)/ $(1)@$(2):~/$(PROJ_PATH)/
endef

clean:
	$(call clean_repo)

k2:
	$(call sync_repo_to,quang,k2)

local:
	$(call sync_repo_to,quang,local)

all: k2 local

fetch_local: