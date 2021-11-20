#include "falk_rng.hh"
#include <cstdio>
int main() {
	rng::distributions::uniform random_dist( 50., .25 );
	for ( int i=5; i-->0; )
		std::printf( "%g\n", rng::get(random_dist) );
}
