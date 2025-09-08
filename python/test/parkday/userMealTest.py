# bring in user bag of words
# bring in bag of words for all recipes
# embed and then find closest predict

## i'm  really not too sure here as the documentation isn't great, but as a first stab
# use trainMode = 1 and just embed one user with ingredients all spaced out
# and everything else as a row of ingredients all spaced out ??? 
# not positive on this one yet.

# one interesting question that arises is if the user bag of words should be fully concatenated as done before
# or if you should restore the meal-level grouping, thus making it more analogous to the overall meal structure?


# perhaps the user doesn't actually need to be embedded - just part of the query for nearest neighbor or prediction?
# for example - the 400 different iterations of the space based on 16k recipes vs 1 usdr at a time won't be ANY different.

# really just need to embed the space once, build the model, and run a separate prediction for eac

# challeng is if the binding is there for python methods or if you need to just use shell commands, or even build out your own sim method (check the paper again)

# look at docspace trainmode = 1
# roger federer loses <tab> venus williams wins <tab> world series ended
# i love cats <tab> funny lolcat links <tab> how to be a petsitter  














# second step, need to tag all meals with which 9 are available that day, and display the closeness ranking for all 9 of the available meals for a day