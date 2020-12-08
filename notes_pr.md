# Notes on PR

TL;DR:
- most features are now vectorized
- added vectorization decorator for custom and un-vectorizable features
- removed eval from feature calculation -> this breaks python code in gsheets
- breaking changes for custom features and people who use the features directly
- 

TODOs before considering merging the PR:
1. There are several breaking changes -> consider version
2. Discuss if and how the vectorization potential across windows could be used (batching, multiprocessing changes etc)
3. The eval removal is optional -> this could be removed if the code injection from gsheets is required


## Vectorized Features
I've switched most features to be implemented with numpys vectorization and added a decorator for feature functions, that cannot be vectorized.

Pros: 
- Calculation with high dimensional data is speed up significantly
- Vectorization decorator makes feature migration a one-liner

Cons:
- **Breaking Datastrcuture change** (see below)
- Features might be less readable at a glance (though i personally think it not too bad)
- In some cases numpy throws an error, as the matrix calculation needs to divide by zero etc, and the matrix can only be masked after the calculation (happy if someone has a better solution)


## Input structure change for feature functions
For this vectorization the *window data structure* needed to be changed. The vectorized features are calculated on the last dimension of the ndarry. This *only* affects people that use the features directly, the structure is reshaped before feature calculation in the `calc_window_features` function. Also, this new would structure allow the features to be calculated at the same time across all channels on all windows. The features hold no assumption about the first n dimensions, only the last one (TODO: check if that also applies for the vectorization decorator).

Pros:
- No assumption of ndarray, except last dimension
- No changes necessary if user uses the feature extraction functions from the library
- Vectorization of features over several windows is free - eg. mean on (.., n windows, x channels) is *one* call
- Vecotrization decorator also handles structure change

Cons:
- **Breaking change** for users, that use the features directly
- The libary currently cannot take full advantage of this due to the way multiprocessing on window the level is used (batching multiple windows would be an option, but increases complexity imensely)


## Removal of eval
I've taken the liberty to remove the eval call in the `calc_window_features` function, for security reasons.

Pros:
- No code injection possible

Cons:
- **Breaking changes** for gsheet support in two ways:
    1. no injection of code like np.arange(0, 10)
    2. parameters with lists should not be strings anymore - ie no {foo: '[0.2, 0.8]'}, but {foo: [0.2, 0.8]})  



