#include <iostream>
#include <experimental/array>
#include "falk_rng.hh"

struct Person { // custom type:
	std::string_view  first_name, family_name;
	bool              is_alive;
	unsigned int      age;
	
	friend std::ostream& operator<<( std::ostream &out, Person const &p ) {
		return out << "{ " << p.first_name << ' ' << p.family_name << ", "
		           << (p.is_alive? "alive":"deceased") << ", age: " << static_cast<int>(p.age) << " }";
	}
};

std::array constexpr family_names = std::experimental::make_array<std::string_view>
	("McCool", "von Stuttgart", "Sanchez", "Leifson", "Chung", "Katamara", "Waktiti", "Wade");
std::array constexpr first_names = std::experimental::make_array<std::string_view>
	("Bob", "Dong", "Ahmed", "Sue", "Alva", "Anastacia", "Urban", "Siri", "Chinkomaru");

namespace rng {
	[[nodiscard]] Person constexpr make_person( RNG_ENGINE(e) ) noexcept { // custom type factory
		return {
			.first_name  = rng::get(first_names,  e), // get random element of range
			.family_name = rng::get(family_names, e), // get random element of range
			.is_alive    = rng::probability(.75, e),
			.age         = rng::get(12u, 80u, e)
		};
	}
}

int main() {
	using namespace rng::literals;
	// creating engines:
	[[maybe_unused]] auto e1 = rng::engine();                 // seeded with a run-time random seed (time-based)
	[[maybe_unused]] auto e2 = rng::engine("this is a seed"); // seeded by string-hashing (C strings, strings_views, std::strings...) 
	[[maybe_unused]] auto e3 = rng::engine(0xDEFEC8);         // seeded with a number
	// using distributions:
	auto constexpr dice_roll = "4d20+69"_d;
	auto constexpr uniform   = rng::uniform_distribution( 0, 100'000 );
	auto constexpr normal    = rng::bounded_normal_distribution( 20.0, 5.0, 4.0 );
	std::cout <<                  "Rolling the dice... " << rng::get( dice_roll, e1 ) << '\n';
	std::cout <<            "Rolling the dice again... " << rng::get( "d20-3"_d, e1 ) << '\n';
	std::cout <<         "Likely a big uniform number: " << rng::get(   uniform, e1 ) << '\n';
	std::cout << "A clamped normal distribution value: " << rng::get(    normal, e1 ) << '\n';
	// alternatively:
	std::cout <<                  "Rolling the dice... " << dice_roll( e1 ) << '\n';
	std::cout <<            "Rolling the dice again... " << "d20-3"_d( e1 ) << '\n';
	std::cout <<         "Likely a big uniform number: " <<   uniform( e1 ) << '\n';
	std::cout << "A clamped normal distribution value: " <<    normal( e1 ) << '\n';
	// using a custom type factory:
	std::cout << "Some random person: " << rng::make_person( e1.fork() ) << '\n'; // stable engine split
	std::cout << "Some random person: " << rng::make_person( e1        ) << '\n';
	// flip a coin:
	std::cout << (rng::flip(e1) ? "Heads!" : "Tails!") << '\n';
	// predict the future:
	if ( rng::probability(.50,e1) )
		std::cout << "It will rain today.\n";
	if ( rng::probability(.33,e1) )
		std::cout << "It will snow today.\n";
	if ( rng::probability(.01,e1) )
		std::cout << "We will all die today.\n";
	// decide on dinner:
	std::cout << "You should eat " << rng::get(    1,  10, e1 ) << " cheese burgers today "
		       <<      "and drink " << rng::get( 4.20, 6.9, e1 ) << " liters of coke.\n";
	// get a number between .0 and 1.0:
	std::cout << "0~1 value: " << rng::random(e1) << '\n';
} 
