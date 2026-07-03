#!/bin/sh
# Test driver: render hello_xr through the WiVRn runtime so the h2-67 encoder
# gets real frames. `sleep infinity |` keeps hello_xr's stdin open so its
# "press any key to quit" thread never fires and it renders continuously.
export XR_RUNTIME_JSON=/home/ramesthegeneric/Projects/WiVRn/build-server/openxr_wivrn-dev.json
exec sh -c 'sleep infinity | hello_xr -g Vulkan2' >/tmp/hello_xr.log 2>&1
