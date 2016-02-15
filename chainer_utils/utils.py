import copy

def get_defaults(setArgs, defArgs, defOnly=True, 
		ignoreKeys=['prmStr']):
	for key in setArgs.keys():
		if defOnly and key not in ignoreKeys:
			assert defArgs.has_key(key), 'Key not found: %s' % key
		defArgs[key] = copy.deepcopy(setArgs[key])
	return defArgs
