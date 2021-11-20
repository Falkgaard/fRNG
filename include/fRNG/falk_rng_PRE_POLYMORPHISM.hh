// rng.hh

// Version: 3.0.0 alpha (WIP)

//  Author: J. Falkgaard (falkgaard@gmail.com) AKA 0xDEFEC8 AKA u/bikki420

// License: MIT
//
// Copyright (c) 2018~2021 J. Falkgaard (falkgaard@gmail.com)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// The algorithms used are public domain. Special thanks to their incredible creators:
//
//    Box-Muller transform by George Edward Pelham Box and Mervin Edgar Muller, pre-1934
//    splitmix64           by Sebastiano Vigna, 2015
//    xoshiro256+          by David Blackman and Sebastiano Vigna, 2018
//    fnv-1a               by Glenn Fowler, Landon Curt Noll, and Kiem-Phong Vo, ca. 1991

// Optional compiler arguments:
//
//   `-D RNG_ENABLE_GLOBAL_ENGINE`       to add a lazily initialized global PRNG engine as a default parameter.
//                                       (IMO it's better to explicitly construct and pass rng::engines by reference).
//
//   `-D RNG_GLOBAL_SEED=<value>`        to give the global engine (if enabled) a persistent, explicit seed.
//                                       valid values for <value>: string literals or 64-bit unsigned integers.
//                                       If not supplied, it will be generated at start-up based off the time.
//                                       For strings, escape the marks thusly: `-D RNG_GLOBAL_SEED=\"seed here\"`
//
// `-D RNG_CT_GENERATE_GLOBAL_SEED`      to generate a fixed global engine seed once every compilation,
//                                       by hashing the date and time of compilation (overrides RNG_GLOBAL_SEED).
//
// `-D RNG_DISABLE_SYNTACTIC_SUGAR`      to remove the `rng::flip(...)` and `rng::probability(p,...)` functions.
//                                       (since they're more-or-less equivalent to `rng::random(...) < .5` and
//                                       `rng::random(...) < p` respectively; might be deemed bloated.)
//
// `-D RNG_DEFAULT_F=<float or double>`  to set the default precision used by functions that can't deduce the
//                                       precision from any parameters. (Note: it can still be overriden by
//                                       explicitly supplying the precision as template arguments, for example:
//                                       `rng::random<double>(...)`. If the compiler argument is not provided,
//                                       the library will default to using floats as the default precision.) 
//
// `-D RNG_DICE_LITERAL=<label>`         to set the literal to use for dice ranges (such as "4d20+3"), e.g.
//                                       `-D RNG_DICE_LITERAL=roll` for `"4d20+3"_roll`. (default: d)
//
// `-D RNG_GLOBAL_ENGINE_LAZY_INIT`      to lazily initialize the the global engine (static function-local variable)
//                                       wrapped with a function instead of using a global variable.
//
// `-D FNV_HASH_LITERAL=<label>`         to set the literal to use for fnv-1a hashing (such as "some_seed"_h), e.g.
//                                       `-D FNV_HASH_LITERAL=hash` for `"text"_hash`. (default: h)

#include <string_view>
#include <concepts>
#include <limits>
#include <chrono>
#include <optional>
#include <cmath>
#include <array>
#include <string>
#include <numbers>
#include <stdexcept>

namespace { // unnamed namespace with implementation helpers
	// just to avoid polluting other files
	using i8  = std::int8_t;
	using i16 = std::int16_t;
	using i32 = std::int32_t;
	using i64 = std::int64_t;
	
	using u8  = std::uint8_t;
	using u16 = std::uint16_t;
	using u32 = std::uint32_t;
	using u64 = std::uint64_t;
	
	using f32 = float;
	using f64 = double;
	
	template <typename T, std::floating_point F> requires( std::integral<T> or std::floating_point<T> )
	[[nodiscard]] auto constexpr lerp( T const min, T const max, F const factor ) noexcept {
		return static_cast<T>( std::fma( factor, max, std::fma(-factor, min, min) ) );
	}
} // end-of-namespace <unnamed>

namespace fnv { // hashing module (TODO: include the full implementation header instead)
	[[nodiscard]] auto constexpr hash( std::string_view const key ) noexcept { // FNV-1a
		u64 constexpr prime { 0x0000'0100'0000'01B3 };
		u64 constexpr basis { 0xCBF2'9CE4'8422'2325 };
		u64 result = basis;
		for ( auto c : key ) {
			result ^= static_cast<u64>(c);
			result *= prime;
		}
		return result;
	}
	
	namespace literals {
		# ifndef FNV_HASH_LITERAL
		#    define FNV_HASH_LITERAL h
		# endif
		// some ugly macros to allow the user to provide the operator's literal at compile-time
		# define FNV_IMPL_HASH_LITERAL_OP_PREFIX  operator""_
		# define FNV_IMPL_CONCAT1(A,B)            FNV_IMPL_CONCAT2(A,B)
		# define FNV_IMPL_CONCAT2(A,B)            A##B
		# define FNV_IMPL_HASH_LITERAL_OP         FNV_IMPL_CONCAT1( FNV_IMPL_HASH_LITERAL_OP_PREFIX, FNV_HASH_LITERAL )
		[[nodiscard]] auto constexpr FNV_IMPL_HASH_LITERAL_OP ( char const * const s, std::size_t const n ) noexcept {
			return hash( std::string_view(s,n) );
		}
		# undef FNV_IMPL_HASH_LITERAL_OP_PREFIX
		# undef FNV_IMPL_CONCAT1
		# undef FNV_IMPL_CONCAT2
		# undef FNV_IMPL_HASH_LITERAL_OP
	} // end-of-namespace fnv::literals
} // end-of-namespace fnv

namespace rng { // TODO: non-unform distributions, range/collection compatibility, enum compatibility	
	# ifndef RNG_DEFAULT_F
	#    define RNG_DEFAULT_F f64
	# endif
	static_assert( std::is_floating_point_v<RNG_DEFAULT_F>, "RNG_DEFAULT_F must be float or double!" );
	using default_precision_floating_point_t = RNG_DEFAULT_F;
	
	namespace { // unnamed namespace (for implementation details)
		[[nodiscard]] bool constexpr is_number( char const c ) noexcept {
			return c >= '0' and c <= '9';
		}
		
		[[nodiscard]] std::pair<char const *,int> constexpr read_integer( char const *s, std::size_t const size ) noexcept {
			bool is_negative = false; // assumed true in case no symbol is provided
			if ( *s == '+' )
				++s; // advance; already assumed true
			else if ( *s == '-' ) {
				is_negative = true;
				++s; // advance
			}
			
			int result = 0;
			for ( std::size_t i=0; i<size and is_number(*s); ++i,++s )
				result = result * 10 + (*s-'0');
			if ( is_negative )
				result *= -1;
			return { s, result };
		}
		
		template <std::size_t N> // TODO: superfluous?
		[[nodiscard]] auto constexpr read_integer( char const(&s)[N] ) noexcept {
			return read_integer(s,N);
		}
	} // end-of-namespace rng::<unnamed>
	
	namespace distributions {
		class dice final {
			u16 m_number_of_dice, m_number_of_sides;
			i32 m_modifier;
		public:
			using value_type = int;
			
			constexpr dice(
				u16 const number_of_dice,
				u16 const number_of_sides,
				i32 const modifier
			):
				m_number_of_dice  { number_of_dice  },
				m_number_of_sides { number_of_sides },
				m_modifier        { modifier        }
			{}
			
			[[nodiscard]] static dice constexpr from_dsl( char const * const dsl_c_str, std::size_t const dsl_c_str_size ) {
				auto s1  = dsl_c_str;
				auto end = dsl_c_str + dsl_c_str_size;
				
				int  number=0, sides=0, mod=0; // values to extract
				
				auto result1          = read_integer(s1, static_cast<std::size_t>(end-s1));
				auto s2               = std::get<0>(result1);
				auto extracted_number = std::get<1>(result1);
				
				if ( s2 != s1 )
					number = extracted_number; // TODO: ensure constraints (>0?)
				else
					number = 1; // no (optional) number of dice supplied
				
				if ( *s2 == 'd' ) [[likely]]
					++s2; // advance past the symbol
				else [[unlikely]]
					throw std::invalid_argument("Invalid format: expected 'd'!");
				
				auto result2         = read_integer(s2, static_cast<std::size_t>(end-s2));
				auto s3              = std::get<0>(result2);
				auto extracted_sides = std::get<1>(result2); // TODO: ensure constraints (>0?)
				
				if ( s3 == s2 ) [[unlikely]]
					throw std::invalid_argument("Invalid format: no number of dice sides supplied!");
				else [[likely]]
					sides = extracted_sides;
				
				if ( s3 == end ) // at end; no (optional) modifier supplied
					mod = 0;
				else {
					auto result3       = read_integer(s3, static_cast<std::size_t>(end-s3));
					auto s4            = std::get<0>(result3);
					auto extracted_mod = std::get<1>(result3); // TODO: ensure constraints
					if ( s4 != s3+1 ) [[likely]]
						mod = extracted_mod;
					else [[unlikely]]
						throw std::invalid_argument("Invalid format: expected modifier!");
					
					if ( s4 != end )
						throw std::invalid_argument("Invalid format: expected end-of-string!");
				}
				
				return { static_cast<u16>(number), static_cast<u16>(sides), mod };
			}
			
			[[nodiscard]] static dice constexpr from_dsl( std::string_view sv ) {
				return from_dsl( sv.data(), sv.size() );
			}
			
			[[nodiscard]] auto constexpr number_of_dice()  const noexcept { return m_number_of_dice;  }
			[[nodiscard]] auto constexpr number_of_sides() const noexcept { return m_number_of_sides; }
			[[nodiscard]] auto constexpr modifier()        const noexcept { return m_modifier;        }
			
			[[nodiscard]] auto constexpr min() const noexcept {
				return +m_number_of_dice + +m_modifier;
			}
			
			[[nodiscard]] auto constexpr max() const noexcept {
				return +m_number_of_dice * +m_number_of_sides + m_modifier;
			}
			
			[[nodiscard]] /*inline?*/ auto to_string() const noexcept {
				std::string result {};
				if ( m_number_of_sides > 0 )
					result += std::to_string(m_number_of_dice);
				result += 'd' + std::to_string(m_number_of_sides);
				if ( m_modifier != 0 )
					result += (m_modifier > 0 ? "+" : "") + std::to_string(m_modifier);
				return result;
			}
			
			template <std::floating_point F = default_precision_floating_point_t>
			[[nodiscard]] F constexpr average() const noexcept {
				return static_cast<F>(.5) * (+m_number_of_sides + 1) * +m_number_of_dice + m_modifier;
			}
		}; // end-of-class rng::distributions::dice
		
		template <typename T> requires( std::integral<T> or std::floating_point<T> )
		class uniform final {
			T m_min, m_max;
		public:
			using value_type = T;
			
			constexpr uniform( T const min, T const max ) noexcept: m_min{ min }, m_max{ max } {}
			
			[[nodiscard]] auto constexpr min() const noexcept {
				return m_min;
			}
			
			[[nodiscard]] auto constexpr max() const noexcept {
				return m_max;
			}
			 
			[[nodiscard]] /*inline?*/ auto to_string() const noexcept {
				std::string result {};
				result += std::to_string(m_min) + '~' + std::to_string(m_max);
				return result;
			}    
			
			template <std::floating_point F = default_precision_floating_point_t>
			[[nodiscard]] auto constexpr average() const noexcept {
				return static_cast<F>(.5) * (m_min + m_max);
			}
		}; // end-of-class rng::distributions::uniform
		
		template <typename            T = default_precision_floating_point_t,
		          std::floating_point F = default_precision_floating_point_t,
					 bool       is_bounded = false>
		requires ( std::integral<T> or std::floating_point<T> )
		class normal final {
		// TODO: replace is_bounded with optional constexpr float?
		//       split into separate templates instead?
		//       depart from immutable API and allow for setters/getters?
		//       cache min/max as members?
		public:
			using          value_type = T; // TODO: ponder!
			using          float_type = F;
			using cache_optional_type = std::optional<float_type>;
		private:
			float_type                  m_mean               = static_cast<float_type>(.0);
			float_type                  m_standard_deviation = static_cast<float_type>(.0);
			float_type                  m_bound_factor       = .0;
			cache_optional_type mutable m_maybe_cache        = std::nullopt;
		public:
			// will set the cache
			inline void cache( float_type const normal ) const noexcept {
				m_maybe_cache = normal;
			}
			
			constexpr normal(
				float_type const mean,
				float_type const standard_deviation
			) noexcept requires( not is_bounded ):
				m_mean               { mean               },
				m_standard_deviation { standard_deviation }
			{}
			
			constexpr normal(
				float_type const mean,
				float_type const standard_deviation,
				float_type const bound_factor
			) noexcept requires( is_bounded ):
				m_mean               { mean               },
				m_standard_deviation { standard_deviation },
				m_bound_factor       { bound_factor       }
			{}
			
			constexpr normal( normal const & ) noexcept = default;
			constexpr normal( normal      && ) noexcept = default;
			~normal() noexcept = default;
			
			// will invalidate the cache (if any) after returning it	
			[[nodiscard]] inline cache_optional_type constexpr maybe_extract_cache() const noexcept {
				cache_optional_type result {};
				m_maybe_cache.swap( result );
				return result;
			}
			
			[[nodiscard]] float_type constexpr mean() const noexcept {
				return m_mean;
			}
			
			[[nodiscard]] float_type constexpr standard_deviation() const noexcept {
				return m_standard_deviation;
			}
			
			[[nodiscard]] auto constexpr min() const noexcept {
				if constexpr ( is_bounded )
					return m_mean - m_standard_deviation * static_cast<float_type>( m_bound_factor );
				else return -std::numeric_limits<F>::infinity();
			}
			
			[[nodiscard]] auto constexpr max() const noexcept {
				if constexpr ( is_bounded )
					return m_mean + m_standard_deviation * static_cast<float_type>( m_bound_factor );
				else return +std::numeric_limits<F>::infinity();
			}
			
			[[nodiscard]] auto constexpr average() const noexcept {
				return m_mean;
			}
			
			[[nodiscard]] auto constexpr maybe_clamp( float_type const v ) const noexcept {
				if constexpr ( is_bounded )
					return std::clamp( v, min(), max() );
				else return v;
			}
			
			[[nodiscard]] auto to_string() const noexcept {
				char buffer[128];
				std::sprintf( buffer, "μ: %g, σ: %g", m_mean, m_standard_deviation );
				return std::string( buffer );
			}
		}; // end-of-class rng::distributions::normal
		
		//template <typename            T = default_precision_floating_point_t,
		//          std::floating_point F = default_precision_floating_point_t>
		//using bounded_normal = normal<T,F,true>;
		
	} // end-of-namespace rng::distributions
	
	// meta-functions (mostly related to distributions):
	namespace meta {
		// uniform:
		template <typename>
		struct is_uniform_distribution final: std::false_type {};
		
		template <typename T>
		struct is_uniform_distribution<distributions::uniform<T>> final: std::true_type {};
		
		template <typename T>
		auto constexpr is_uniform_distribution_v = is_uniform_distribution<T>::value;
		
		// normal:
		template <typename>
		struct is_normal_distribution final: std::false_type {};
		
		template <typename...Ts>
		struct is_normal_distribution<distributions::normal<Ts...>> final: std::true_type {};
		
		template <typename T>
		auto constexpr is_normal_distribution_v = is_normal_distribution<T>::value;
		
		// dice:
		template <typename T>
		auto constexpr is_dice_distribution_v = std::is_same_v<distributions::dice,T>;
		
		// any:
		template <typename T>
		auto constexpr is_distribution_v = is_dice_distribution_v<T>
		                                or is_uniform_distribution_v<T>
		                                or is_normal_distribution_v<T>;

/* TODO:
	
	template <typename T, typename A>
	class distribution {
	public:
		using value_type   = T;
		using average_type = A; // TODO: better name
		
		virtual ~distribution() noexcept = default;
		
		[[nodiscard]] virtual T constexpr min()                                      const noexcept = 0;
		[[nodiscard]] virtual T constexpr max()                                      const noexcept = 0;
		[[nodiscard]] virtual A constexpr average()                                  const noexcept = 0;
		[[nodiscard]] virtual T constexpr get( engine &e OR_OPTIONAL_GLOBAL_ENGINE ) const noexcept = 0;
	};
	
	template <std::integral I, std::floating_point F = default_floating_point_precision>
	class dice_distribution final : public distribution<I,F> {
		I m_number_of_dice;
		I m_number_of_sides;
		I m_modifier;
	public:
		// Constructors etc
		// from_dsl, getters, setters
		[[nodiscard]] I constexpr min()                                      const override final noexcept;
		[[nodiscard]] I constexpr max()                                      const override final noexcept;
		[[nodiscard]] F constexpr average()                                  const override final noexcept;
		[[nodiscard]] I constexpr get( engine &e OR_OPTIONAL_GLOBAL_ENGINE ) const override final noexcept;
	};
	
	template <typename T, std::floating_point F = default_floating_point_precision>>
	requires( std::integral<T> or std::floating_point<T> )
	class uniform_distribution final : public distribution<T,F> {
	protected:
		T m_min;
		T m_max;
	public:
		// Constructors etc
		// getters, setters
		[[nodiscard]] T constexpr min()                                      const override final noexcept;
		[[nodiscard]] T constexpr max()                                      const override final noexcept;
		[[nodiscard]] F constexpr average()                                  const override final noexcept;
		[[nodiscard]] T constexpr get( engine &e OR_OPTIONAL_GLOBAL_ENGINE ) const override final noexcept;
	};
	
	template <typename T, std::floating_point F = default_floating_point_precision>>
	requires( std::integral<T> or std::floating_point<T> )
	class normal_distribution : public distribution<T,F> {
		F                        m_mean;
		F                        m_standard_deviation;
		std::optional<F> mutable m_cache;
	public:
		// getters, setters
		[[nodiscard]] virtual T constexpr min()                              const noexcept = 0;
		[[nodiscard]] virtual T constexpr max()                              const noexcept = 0;
		[[nodiscard]] F constexpr average()                                  const override final noexcept;
		[[nodiscard]] T constexpr get( engine &e OR_OPTIONAL_GLOBAL_ENGINE ) const override final noexcept;
	};
	
	template <typename T, std::floating_point F = default_floating_point_precision>>
	requires( std::integral<T> or std::floating_point<T> )
	class bounded_normal_distribution final : public normal_distribution<T,F> {
	public:
		// Constructors etc
		// getters, setters
		[[nodiscard]] T constexpr min() const override final noexcept;
	[[nodiscard]] T constexpr max() const override final noexcept;
	};
	
	template <typename T, std::floating_point F = default_floating_point_precision>>
	requires( std::integral<T> or std::floating_point<T> )
	class unbounded_normal_distribution final : public normal_distribution<T,F> {
		F m_bound_factor;
	public:
		// Constructors etc
		// getters, setters
		[[nodiscard]] T constexpr min() const override final noexcept;
		[[nodiscard]] T constexpr max() const override final noexcept;
	};
*/
		
		// concepts:
		template <typename T> concept a_dice_distribution    = is_dice_distribution_v<T>;
		template <typename T> concept a_uniform_distribution = is_uniform_distribution_v<T>;
		template <typename T> concept a_normal_distribution  = is_normal_distribution_v<T>;
		template <typename T> concept a_distribution         = is_distribution_v<T>;
	} // end-of-namespace rng::meta
	
	namespace literals {
		# ifndef RNG_DICE_LITERAL
		#    define RNG_DICE_LITERAL d
		# endif
		// some ugly macros to allow the user to provide the operator's literal at compile-time
		# define RNG_IMPL_DICE_LITERAL_OP_PREFIX  operator""_
		# define RNG_IMPL_CONCAT1(A,B)            RNG_IMPL_CONCAT2(A,B)
		# define RNG_IMPL_CONCAT2(A,B)            A##B
		# define RNG_IMPL_DICE_LITERAL_OP         RNG_IMPL_CONCAT1( RNG_IMPL_DICE_LITERAL_OP_PREFIX, RNG_DICE_LITERAL )
		[[nodiscard]] constexpr auto RNG_IMPL_DICE_LITERAL_OP ( char const * const s, std::size_t const n ) {
			return distributions::dice::from_dsl( s, n );
		}
		# undef RNG_IMPL_DICE_LITERAL_OP_PREFIX
		# undef RNG_IMPL_CONCAT1
		# undef RNG_IMPL_CONCAT2
		# undef RNG_IMPL_DICE_LITERAL_OP
	} // end-of-namespace rng::literals
	
	class seed final { // TODO: copy & move methods
			u64                              m_value;
			std::optional<std::string_view>  m_maybe_string_source = {};
		public:
			using clock = std::chrono::high_resolution_clock;
			
			seed() noexcept: // default construct with a time-based random value
				m_value { static_cast<u64>(clock::now().time_since_epoch().count()) }
			{}
			
			// take an explicit internal value (e.g. during deserialization)
			explicit constexpr seed( u64 const n ) noexcept:
				m_value { n }
			{}
			
			// convert a string_view to an internal value (via hashing)
			explicit constexpr seed( std::convertible_to<std::string_view> auto &&sv ) noexcept:
				m_value               { fnv::hash(std::forward<decltype(sv)>(sv)) },
				m_maybe_string_source { sv }
			{}
			
			[[nodiscard]] auto constexpr value()               const noexcept { return m_value; }
			[[nodiscard]] auto constexpr maybe_string_source() const noexcept { return m_maybe_string_source; }
	}; // end-of-class rng::seed
	
	class engine final { // Xoshiro256+
		std::array<u64,4> m_state; // 256-bit internal entropy state
	public:
		explicit constexpr engine( seed s = seed{} ) noexcept {
			auto t = s.value();
			for ( u64 &e: m_state ) { // initializing state with seed + splitmix64
				e =             (t += 0x9E37'79B9'7F4A'7C15);
				e = (e ^ (e >> 30)) * 0xBF58'476D'1CE4'E5B9;
				e = (e ^ (e >> 27)) * 0x94D0'49BB'1331'11EB;
				e = (e ^ (e >> 31));
			}
		}
		
		// `fork()` mutates the state once (like next) and returns a new engine.
		// when generator stability is important, e.g. you want to use your seed's random sequence
		// in a persistent manner but you also want to be able to make additions (or subtractions)
		// in branch generations, then the best approach is to pass those functions a fork of your
		// engine instead of a reference. There's a demo at the bottom.
		[[nodiscard]] engine constexpr fork() noexcept {
			return engine( seed{ next() } );
		}
		
		u64 constexpr next() noexcept {
			u64 const result = m_state[0] + m_state[3];
			u64 const temp   = m_state[1] << 17;
			m_state[2] ^= m_state[0];
			m_state[3] ^= m_state[1];
			m_state[1] ^= m_state[2];
			m_state[0] ^= m_state[3];
			m_state[2] ^= temp;
			m_state[3]  = (m_state[3]<<45) | (m_state[3]>>19); // rotate left
			return result;
		}
	}; // end-of-class rng::engine
	
	// some pre-processor stuff for providing an opt-in global engine as well as some configuration options:
	#ifdef RNG_ENABLE_GLOBAL_ENGINE
		#ifdef RNG_CT_GENERATE_GLOBAL_SEED
			seed constexpr global_seed { fnv::hash(__DATE__) xor fnv::hash(__TIME__) }; // compile-time generated seed
		#elif defined(RNG_GLOBAL_SEED)
			seed constexpr global_seed{ ( RNG_GLOBAL_SEED ) }; // optional compile-time seed
		#else // let seed default construct at run-time start-up using <chrono>
			seed const global_seed {};
		#endif
		#ifdef RNG_GLOBAL_ENGINE_LAZY_INIT
			// use a lazily initialized global engine wrapped in a function
			[[nodiscard]] inline engine& global_engine() noexcept {
				static engine e { global_seed }; // constructed the first time the function gets called
				return e;
			}
			#define OR_OPTIONAL_GLOBAL_ENGINE = rng::global_engine()
		#else // use a global variable
			engine global_engine { global_seed }; // constructed when the program starts
			#define OR_OPTIONAL_GLOBAL_ENGINE = rng::global_engine
		#endif
	#else // disable global engine default
		#define OR_OPTIONAL_GLOBAL_ENGINE
	#endif
	
	// returns a floating point value within the range (0,1)
	template <std::floating_point F = default_precision_floating_point_t>
	[[nodiscard]] F constexpr random( engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
		return static_cast<F>(e.next()) / static_cast<F>(std::numeric_limits<u64>::max());
	}
	
	// returns a value within the (min,max) range provides by the caller; tries to deduce the type
	// but this can be explicitly overrided by providing the type explictly as a template parameter, e.g:
	// `rng::get<float>(0,1)` would implicitly cast the parameters to floats and use them to generate a float.
	template <typename T, std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	[[nodiscard]] T constexpr get( T const min, T const max, engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
		return lerp( min, max, random<F>(e) );
	}
	
	// returns a value between the min and max values of a type
	// e.g. `rng:get<std::uint8_t>(/*optional engine ref*/)` would return a byte with a value between 0~255
	template <typename T, std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	[[nodiscard]] T constexpr get( engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
		return lerp( std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), random<F>(e) );
	}
	
	// takes a dice range (e.g 2d8+5, 3d3, d9-5, etc; literal version is recommended) and returns a random rolled int
	template <std::floating_point F = default_precision_floating_point_t>
	[[nodiscard]] auto constexpr get( distributions::dice const &d, engine &e OR_OPTIONAL_GLOBAL_ENGINE ) {
		auto result = d.modifier();
		for ( auto i=0; i<d.number_of_dice(); ++i )
			result += get( static_cast<u16>(0), d.number_of_sides(), e ) + 1;
		return result;
	}
	
	// returns a value within the (min,max) distributions::uniform provided by the caller
	template <std::floating_point F = default_precision_floating_point_t>
	[[nodiscard]] auto constexpr get( meta::a_uniform_distribution auto const &d, engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
		return lerp( d.min(), d.max(), random<F>(e) );
	}
	
	// returns a normal distributed value (mean, standard deviation, floating point precision, and target type are provided with argument d
	template <meta::a_normal_distribution D>
	[[nodiscard]] auto constexpr get( D const &d, engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
		using F = typename D::float_type;
		//using V = typename D::value_type;
		
		F constexpr epsilon = std::numeric_limits<F>::epsilon();
		F constexpr pi_x2   = static_cast<F>(2.) * std::numbers::pi_v<F>;
		
		F result;
		
		auto maybe_cache = d.maybe_extract_cache(); // will invalidate any existing cache in d
		if ( maybe_cache )
			result = maybe_cache.value();
		else {
			F uniform1, uniform2;
			do {
				uniform1 = rng::random(e);
				uniform2 = rng::random(e);
			}
			while ( epsilon >= uniform1 );
			// use the Box-Muller transform to convert (.0~1.) uniform values to normal distribution values
			F const magnitude = std::sqrt( std::log(uniform1) * static_cast<F>(-2.) ) * d.standard_deviation();
			d.cache( magnitude * std::cos( pi_x2 * uniform2 ) + d.mean() );
			result = magnitude * std::sin( pi_x2 * uniform2 ) + d.mean();
		}
		
		return d.maybe_clamp(result);
	}
	
	#ifndef RNG_DISABLE_SYNTACTIC_SUGAR
		// returns true p% of the time
		[[nodiscard]] bool constexpr probability( std::floating_point auto const p, engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
			return random(e) < p;
		}
		
		// coinflip; returns true 50% of the time
		[[nodiscard]] bool constexpr flip( engine &e OR_OPTIONAL_GLOBAL_ENGINE ) noexcept {
			return static_cast<std::int64_t>(e.next()) < 0; // using sign bit
		}
	#endif
} // end-of-namespace rng

