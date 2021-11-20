#include "falk_rng.hh"
#include <iostream>
#include <experimental/array>

// custom type (+ data):
constexpr std::array family_names = std::experimental::make_array<std::string_view>(
   "McCool", "von Stuttgart", "Sanchez", "Leifson"
);
constexpr std::array first_names = std::experimental::make_array<std::string_view>(
   "Bob", "Dong", "Ahmed", "Sue", "Alva", "Anastacia", "Urban", "Siri"
);
struct Person {
   std::string_view  first_name, family_name;
   bool              is_alive;
   unsigned int      age;

   friend std::ostream& operator<<( std::ostream &out, Person const &p ) {
      return out << "{ " << p.first_name << ' ' << p.family_name << ", "
                 << (p.is_alive? "alive":"deceased") << ", age: " << static_cast<int>(p.age) << " }";
   }
};

// rng extension to support the custom type:
// feel free to use the rng namespace (preferably with a `make_<class-name>(...)` scheme)
// *however* I can't guarantee no collisions in later versions then, *but* they should be astronomically unlikely.
// also, I recommend having `rng::engine &e OR_OPTIONAL_GLOBAL_ENGINE` as the last parameter for stylistic consistency.
namespace rng {
   // note: you can add additional parameters before the engine and use them during the generation
   //       as well as create various variants with different parameter sets (including tag dispatch).
   [[nodiscard]] Person constexpr make_person( rng::engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
      return {
         .first_name  = first_names[  rng::get(static_cast<std::size_t>(0), std::size(first_names),  e) ],
         .family_name = family_names[ rng::get(static_cast<std::size_t>(0), std::size(family_names), e) ],
         .is_alive    = rng::flip(e),
         .age         = rng::get(12u, 80u, e)
      };
   }
} // end-of-namespace rng

#define ACTIVE_PROGRAM 3 // set the active main() here

#if ACTIVE_PROGRAM == 0 // global engine example (also showcasing various features):

   int main() {
      using namespace rng::literals;
      std::cout <<                         "d100 = " <<  rng::get(1,100)                        << '\n'
                <<                "random person = " <<  rng::make_person()                     << '\n'
                <<                "\"4d20+69\"_d = " <<  rng::roll("4d20+69"_d)                 << '\n'
                << "random unit factor (0.0~1.0) = " <<  rng::random()                          << '\n'
                <<                     "4.20~6.9 = " <<  rng::get(4.20,6.9)                     << '\n'
                <<    "Should I drink more coffee? " << (rng::flip()           ? "yes" : "no" ) << '\n'
                <<        "Should I procrastinate? " << (rng::probability(.90) ? "yes" : "no" ) << '\n';
      
      #ifdef RNG_ENABLE_GLOBAL_ENGINE
         std::cout << "Global seed: " << rng::global_seed // contains the global engine's seed (if the global engine is enabled)
                                         .maybe_string_source() // returns a std::optional<std::string_view> for source string
                                         .value_or( std::to_string( rng::global_seed.value() ) ) << '\n'; // fallback to actual integer
      #endif
   }

#elif ACTIVE_PROGRAM == 1 // generated assembly comparison between hard-coded values and seeded generated values:

   int main() { 
      rng::engine e {"optional explicit seed string or number here"}; // changing this will naturally change the values
      std::cout <<                       "d100 =" <<  rng::get(1,100,e)           << '\n'
                <<                  "4.20~6.9 = " <<  rng::get(4.20,6.9,e)        << '\n'
                << "Should I drink more coffee? " << (rng::flip(e)? "yes" : "no") << '\n';
   }

#elif ACTIVE_PROGRAM == 2 // the generated assembly is the same either way -->

   int main() { // assumes that the seed "optional explicit seed string or number here" is used above
      std::cout <<                       "d100 =" << 32                   << '\n'
                <<                  "4.20~6.9 = " << 6.314832866191864    << '\n'
                << "Should I drink more coffee? " << (true? "yes" : "no") << '\n';
   }

#elif ACTIVE_PROGRAM == 3

   #include <iomanip>
   #include <algorithm>
	#include <span>
   #include <ranges>
	#include <cassert>

   template <typename T>
   [[nodiscard]] std::vector<double>
	compute_distribution( std::vector<T> const &data, std::size_t const sum ) {
		auto const size = std::size(data);
		// TODO: use transform, make sum optional (compute if not provided)
      std::vector<double> result( size );
      for ( auto i=0ull; i<size; ++i )
         result[i] = static_cast<double>(data[i]) / static_cast<double>(sum);
      return result;
   }

   // constraints: N<100
   void draw_histogram( std::vector<double> const &distribution_data, std::size_t rows=10 ) {
		auto const size = std::size(distribution_data);
      assert( size < 100 and "Bottom index printing logic only supports size<100!" );
      auto const max  = *std::ranges::max_element( distribution_data );
      for ( auto row=0ull; row<rows; ++row ) {
         auto const threshold = max * ( 1.0 - static_cast<double>(row) / static_cast<double>(rows) );
         for ( auto i=0ull; i<size; ++i )
            std::cout << ' ' << (distribution_data[i] >= threshold? "█":"·");
         std::cout << " │ " << std::fixed << std::setprecision(4) << threshold << "\n";
      }
      // print bottom separator + indices:
      for ( int i=0; i<static_cast<int>(size); ++i )
         std::cout << "══";
      std::cout << "═╧═══════\n";
      for ( int i=0; i<static_cast<int>(size); ++i )
         std::printf( " %c", '0' + (i < 10? i : (i/10) ) );
      std::cout << '\n';
      for ( int i=0; i<static_cast<int>(size); ++i )
         std::printf( " %c", (i < 10? ' ' : '0'+(i%10) ) );
      std::cout << "\n\n";
   }

	template <std::size_t N, typename...Args>
	void simulate( rng::distribution<Args...> const &d, std::size_t const rolls, rng::engine &e OR_OPTIONAL_GLOBAL_ENGINE ) {
		auto const min_value = static_cast<std::size_t>( d.min() );
		auto const max_value = static_cast<std::size_t>( d.max() );
		assert( min_value >= 0 );
		assert( max_value <= N );
		std::vector<std::uint64_t> stats(N);
		for ( auto iteration=0u; iteration<rolls; ++iteration )
			++stats[static_cast<std::uint64_t>(rng::get(d,e))];
		// for ( auto i=0U; i<N; ++i )
		// 	std::cout << std::setw(2) << i << ": " << stats[i] << '\n';
		auto distribution_data = compute_distribution( stats, rolls );
		draw_histogram( distribution_data );
	}

   int main() {
      using namespace rng::literals;
		auto e = rng::engine{};
		auto  constexpr max_value   = 41;
      auto  constexpr total_rolls = 1'000'000ull;

      std::cout << "\nPRNG Distribution Test Histogram(s) with " << total_rolls << " rolls each:\n\n";
#if 1
      auto  constexpr dice        = "20d3-20"_d; // type is rng::distributions::dice
      std::cout << dice.to_string() << ":\n";
      simulate<max_value>(   dice, total_rolls, e );

		auto  constexpr linear      = rng::uniform_distribution        {    0, max_value };
      std::cout << linear.to_string() << ":\n";
      simulate<max_value>( linear, total_rolls, e );
#endif	
 		auto  constexpr normal      = rng::bounded_normal_distribution<unsigned int> { 20., 5., 3.5 };
      std::cout << normal.to_string() << ":\n";
      simulate<max_value>( normal, total_rolls, e );
   }

#endif




// One thing that's not covered in the demos above is engine forking for stable random generation:
#if 0
   class some_complex_type final {/*...*/};
   [[nodiscard]] auto generate_some_complex_type( /*... ,*/ rng::engine e ) { // engine taken by value or rvalue reference
      some_complex_type result;
      // use rng functions with e to initialize the state of result
      return result;
   }

   int main() {
      rng::engine e { /*seed:*/ 12345 };
      auto v = generate_some_complex_type(/*... ,*/ e.fork() ); // result of fork passed
      auto n = rng::get(1, 100, e); // <- the result of this call won't change even if the number of rng calls in the
   }                                //    `generate_some_complex_type(...)` function changes. E.g. new members added.
#endif
