import copy

def get_defaults(setArgs, defArgs, defOnly=True):
	for key in setArgs.keys():
		if defOnly:
			assert defArgs.has_key(key), 'Key not found: %s' % key
		defArgs[key] = copy.deepcopy(setArgs[key])
	return defArgs
