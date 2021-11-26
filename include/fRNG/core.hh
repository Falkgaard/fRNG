// falk_rng.hh

// Version: 3.0.0 alpha (WIP)
//
//  Author: Falkgaard (falkgaard@gmail.com) AKA 0xDEFEC8 AKA u/bikki420
//
// License: MIT
//
// Copyright (c) 2018~2021 J.L.V. Falkengaard (falkgaard@gmail.com)
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
//
// Author's addendum: also, please include the following special thanks as well):
//
// The algorithms used are public domain. Special thanks to their incredible creators:
//
//    Box-Muller transform by George Edward Pelham Box and Mervin Edgar Muller, pre-1934
//    splitmix64           by Sebastiano Vigna, 2015
//    xoshiro256+          by David Blackman and Sebastiano Vigna, 2018
//    fnv-1a               by Glenn Fowler, Landon Curt Noll, and Kiem-Phong Vo, ca. 1991

// Optional compiler arguments:
//
// `-D RNG_ENABLE_GLOBAL_ENGINE`         to add a lazily initialized global PRNG engine as a default parameter.
//                                       (IMO it's better to explicitly construct and pass rng::engines by reference).
//
// `-D RNG_DISABLE_INIT_JUMP`            unless this flag is set, each engine will perform one call to engine::jump()
//                                       during their construction. If your code base creates a lot of engines, then
//                                       this might result in a slight performance boost at the cost of a potentially
//                                       negligible random output quality hit.
//
// `-D RNG_GLOBAL_SEED=<value>`          to give the global engine (if enabled) a persistent, explicit seed.
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
// `-D RNG_DEFAULT_I=<...>`              TODO
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

// TODO: make fnv-hash conditional (opt-out) and allow the user to provide their hash function of choice
//       implement other engines
//       add support for ranges/iterators? e.g. random element in a collection
//       add partial support for enums? (probably best delayed until we have better enums or reflectionn)
//       add rng::get<UserType>(e) for types with PRNG factories?
//       add support for more complex expressions in the dice DSL?
//       polish code and cover more edge cases
//       add compiler flag for naming style? (i.e. uniform_distribution VS. UniformDistribution)
//       make rng::engines threadsafe by using a mutex in rng::engine.next()?
//       utilize SIMD intrinsics?
//       static assert that the hashes of FNV_HASH_LITERAL and RNG_DICE_LITERAL are distinct
//       improve multi-threaded output (goal: same output regardless of number of threads used!)

// NOTE: While the global engine *is* thread-safe, it's easy to lose the determinism in a threaded environment
//       when you use it because different numbers of threads can be assigned different tasks and in different
//       execution orders between run-times. So when reproducibility is desired, it is recommended instead to
//       do some more granular paralleism with thread-local non-global generators. Alternatively using a shared
//       generator called in a critical section (e.g. mutex guarded).
//
//       Whenever possibly, try to split up threads by separate tasks instead of of splitting up a single task
//       into multiple threads. (E.g. have one thread generate things of type X while another generates things of type Y).
//       This way, it's much easier to ensure determinism within each task since you can explicitly give it the same
//       initial generator regardless of which thread performs the task.
//
//       Since the normal_distributions contain a mutable data member cache, to ensure thread-safety you should not 
//       share instances between threads. Instead, create a thread-private copy on the stack of each thread that needs it.

#include <string_view>
#include <concepts>
#include <limits>
#include <chrono>
#include <optional>
#include <cmath>
#include <cassert>
#include <array>
#include <ranges>
#include <string>
#include <numbers>
#include <stdexcept>
#include <omp.h>

#define FALK_FWD( X )               std::forward<decltype((X))>((X))
#define FALK_CONCATENATE(A,B)       FALK_CONCATENATE_IMPL(A,B)
#define FALK_CONCATENATE_IMPL(A,B)  A ## B

namespace { // unnamed namespace with implementation helpers
	namespace details { // refactor
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
	}
} // end-of-unnamed-namespace

namespace fnv { // hashing module
	namespace { // unnamed
		using namespace details; // refactor
	}
	
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
		[[nodiscard]] auto constexpr FALK_CONCATENATE( operator""_ , FNV_HASH_LITERAL ) ( char const * const s, std::size_t const n ) noexcept( noexcept( hash(std::string_view(s,n)) ) ) {
			return hash( std::string_view(s,n) );
		}
	} // end-of-namespace fnv::literals
} // end-of-namespace fnv

namespace rng { // TODO: non-unform distributions, range/collection compatibility, enum compatibility	
	namespace { // unnamed
		using namespace details;
	}
	
	# ifndef RNG_DEFAULT_I
	#    define RNG_DEFAULT_I std::int32_t
	# endif
	static_assert( std::signed_integral<RNG_DEFAULT_I>, "RNG_DEFAULT_I must be a signed integer type!" );
	using default_width_signed_integral_t = RNG_DEFAULT_I;
	
	# ifndef RNG_DEFAULT_F
	#    define RNG_DEFAULT_F double
	# endif
	static_assert( std::floating_point<RNG_DEFAULT_F>, "RNG_DEFAULT_F must be float or double!" );
	using default_precision_floating_point_t = RNG_DEFAULT_F;
	
	namespace { // unnamed namespace (for implementation details)
		[[nodiscard]] bool constexpr is_number( char const c ) noexcept {
			return c >= '0' and c <= '9';
		}
		
		// TODO: use string_view and remove prefix instead
		[[nodiscard]] std::pair<char const *,int> constexpr read_integer( char const *s, std::size_t const size ) noexcept {
			bool is_negative = false; // assumed false in case no symbol is provided
			if ( *s == '+' )
				++s; // advance; already assumed false
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
	
	// A seed class that maps an integer, string, or point in time to a specific integral value
	// that may then be used as a seed value for things such as PRNG engines.
	class seed final { // TODO: copy & move methods?
			u64                             m_value;
			std::optional<std::string_view> m_maybe_string_source = {};
		public:
			using clock = std::chrono::high_resolution_clock;
			
			seed() noexcept: // default construct with a time-based random value
				m_value { static_cast<u64>(clock::now().time_since_epoch().count()) }
			{}
			
			// implicit conversion constructor; take an explicit internal integer value
			constexpr seed( u64 const n ) noexcept:
				m_value { n }
			{}
			
			//  implicit conversion constructor; convert a string value to an internal value (via hashing)
			//  TODO: string_view by value instead?
			constexpr seed( std::convertible_to<std::string_view> auto &&sv ) noexcept:
				m_value               { fnv::hash(std::forward<decltype(sv)>(sv)) },
				m_maybe_string_source { sv }
			{}
			
			// deserialization constructor
			constexpr seed( u64 const v, std::optional<std::string_view> sv ) noexcept:
				m_value               { v  },
				m_maybe_string_source { sv }
			{}
			
			[[nodiscard]] auto constexpr value()               const noexcept { return m_value; }
			[[nodiscard]] auto constexpr maybe_string_source() const noexcept { return m_maybe_string_source; }
	}; // end-of-class rng::seed
	
	// TODO: enable opt-in intrinsics (AVX-2 SIMD) usage;
	//       would generate 4 valid states at once
	//       at the cost of 1024-bit state instead of 256-bits
	//       this way it would only need to mutate the state on every 4th call
	//       maybe add AVX-512 and SSE support as well?
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
			#ifndef RNG_DISABLE_INIT_JUMP
				jump(); // to ensure the initial state is thoroughly mixed
			#endif
		}
		
		// `fork()` mutates the state once (just like a call to next) and returns a new engine.
		// When generator stability is important, e.g. you want to use your seed's random sequence
		// in a persistent manner but you also want to be able to make additions (or subtractions)
		// in branch generations, then the best approach is to pass those functions a fork of your
		// engine instead of a reference.
		[[nodiscard]] engine constexpr fork() noexcept {
			return engine( next() );
		}
		
		// jumps ahead state by 2^128 steps
		void jump() {
			std::array<u64,4> new_state {};
			u64 constexpr jmp_tbl[] {
				0x180E'C6D3'3CFD'0ABA,
				0xD5A6'1266'F0C9'392C,
				0xA958'2618'E03F'C9AA,
				0x39AB'DC45'29B1'661C
			};
			for ( std::size_t i=0; i < std::size(jmp_tbl); ++i ) {
				for ( std::size_t b=0; b < 64; ++b ) {
					if ( jmp_tbl[i] & 1ull << b ) {
						new_state[0] ^= m_state[0];
						new_state[1] ^= m_state[1];
						new_state[2] ^= m_state[2];
						new_state[3] ^= m_state[3];
					}
					next();	
				}
			}
			std::swap( m_state, new_state );
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
	// TODO: improve thread-local seeds...
	#ifdef RNG_ENABLE_GLOBAL_ENGINE
		#ifdef RNG_CT_GENERATE_GLOBAL_SEED
			seed constexpr global_seed { fnv::hash(__DATE__) xor fnv::hash(__TIME__) }; // compile-time generated seed
		#elif defined(RNG_GLOBAL_SEED)
			seed constexpr global_seed{ ( RNG_GLOBAL_SEED ) }; // optional compile-time seed
		#else // let seed default construct at run-time start-up using <chrono>
			seed const global_seed {};
		#endif
		thread_local const seed thread_local_global_seed { (global_seed.value() << omp_get_thread_num()) xor global_seed.value() };
		#ifdef RNG_GLOBAL_ENGINE_LAZY_INIT
			// use a lazily initialized global engine wrapped in a function
			[[nodiscard]] inline engine& global_engine() noexcept {
				static thread_local engine e { thread_local_global_seed }; // constructed the first time the function gets called
				return e;
			}
			#define OR_OPTIONAL_GLOBAL_ENGINE = ::rng::global_engine()
		#else // use a global variable
			static thread_local engine global_engine { thread_local_global_seed }; // constructed when the program starts
			#define OR_OPTIONAL_GLOBAL_ENGINE = ::rng::global_engine
		#endif
	#else // disable global engine default
		#define OR_OPTIONAL_GLOBAL_ENGINE
	#endif
	
	//? template <typename T>
	//? concept some_engine = std::is_same_v<std::remove_cvref_t<T>, rng::engine>; // TODO: expand when more engines are added
	//?	
	//? #define RNG_ENGINE( IDENTIFIER )   rng::some_engine auto && IDENTIFIER OR_OPTIONAL_GLOBAL_ENGINE
	#define RNG_ENGINE( IDENTIFIER )   ::rng::engine &e OR_OPTIONAL_GLOBAL_ENGINE
	
	// returns a floating point value within the range (0,1)
	template <std::floating_point F = default_precision_floating_point_t>
	[[nodiscard]] F constexpr random( RNG_ENGINE(e) ) noexcept {
		return static_cast<F>(e.next()) / static_cast<F>(std::numeric_limits<u64>::max());
	}
	
	// returns a value within the (min,max) range provides by the caller; tries to deduce the type
	// but this can be explicitly overridden by providing the type explictly as a template parameter, e.g:
	// `rng::get<float>(0,1)` would implicitly cast the parameters to floats and use them to generate a float.
	template <typename T, std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	[[nodiscard]] T constexpr get( T const min, T const max, RNG_ENGINE(e) ) noexcept {
		return lerp( min, max, random<F>(FALK_FWD(e)) );
	}
	
	// returns a value between the min and max values of a type
	// e.g. `rng:get<std::uint8_t>(/*optional engine ref*/)` would return a byte with a value between 0~255
	template <typename T, std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	[[nodiscard]] T constexpr get( RNG_ENGINE(e) ) noexcept {
		return lerp( std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), random<F>(FALK_FWD(e)) );
	}
	
	#ifndef RNG_DISABLE_SYNTACTIC_SUGAR
		// returns true p% of the time
		template <std::floating_point F = default_precision_floating_point_t>
		[[nodiscard]] bool constexpr probability( std::floating_point auto const p, RNG_ENGINE(e) ) noexcept {
			return random<F>(FALK_FWD(e)) < p;
		}
		
		// coinflip; returns true 50% of the time
		[[nodiscard]] bool constexpr flip( RNG_ENGINE(e) ) noexcept {
			return static_cast<std::int64_t>(e.next()) < 0; // using sign bit
		}
	#endif
	
// === === === === === === === === === [ distribution module start ] === === === === === === === === === //
	
	// abstract base-class for all distributions
	template <typename T, typename A=T>
	class distribution {
	public:
		using value_type   = T;
		using average_type = A; // TODO: better name
		virtual ~distribution() noexcept = default;
		[[nodiscard]] virtual T min()                      const noexcept = 0; // abstract
		[[nodiscard]] virtual T max()                      const noexcept = 0; // abstract
		[[nodiscard]] virtual A average()                  const noexcept = 0; // abstract
		[[nodiscard]] virtual std::string to_string()      const noexcept = 0; // abstract
		[[nodiscard]] inline T operator()( RNG_ENGINE(e) ) const noexcept { return this->get( FALK_FWD(e) ); }
		[[nodiscard]] inline T operator()( engine &&e )    const noexcept { return this->get( FALK_FWD(e) ); }
	private:
		[[nodiscard]] virtual T get( engine && ) const noexcept = 0; // abstract
	};
	
	template <std::signed_integral I = default_width_signed_integral_t,
	          std::floating_point  F = default_precision_floating_point_t>
	class dice_distribution final : public distribution<I,F> {
		I m_number_of_dice;
		I m_number_of_sides; // TODO: use unsigned type?
		I m_modifier;
	public:
		~dice_distribution() noexcept final = default;
		constexpr dice_distribution(
			I const number_of_dice  = 0,
			I const number_of_sides = 0,
			I const modifier        = 0
		) noexcept:
			m_number_of_dice  { number_of_dice  },
			m_number_of_sides { number_of_sides },
			m_modifier        { modifier        }
		{
			assert( m_number_of_sides >= 0 ); // it doesn't make any rational sense to have a negative number of sides
		}
		[[nodiscard]] static dice_distribution constexpr from_dsl( char const * const dsl_c_str, std::size_t const dsl_c_str_size ) {
		// TODO: use string_view and remove prefix instead of char pointers
			auto s1  = dsl_c_str;
			auto end = dsl_c_str + dsl_c_str_size;
			
			dice_distribution result;
			
			auto const &[s2, extracted_number] = read_integer( s1, static_cast<std::size_t>(end-s1) );
			
			if ( s2 != s1 ) [[unlikely]]
				result.m_number_of_dice = extracted_number; // TODO: ensure constraints (>0?)
			else [[likely]]
				result.m_number_of_dice = 1; // no (optional) number of dice supplied
			
			if ( *s2 != 'd' ) [[unlikely]]
				throw std::invalid_argument("Invalid format: expected 'd'!");
			
			// s2+1 to skip past the 'd'
			auto const &[s3, extracted_sides] = read_integer( s2+1, static_cast<std::size_t>(end-s2) ); // TODO: ensure constraints (>0?)
			
			if ( s3 == s2 ) [[unlikely]]
				throw std::invalid_argument("Invalid format: no number of dice sides supplied!");
			else [[likely]]
				result.m_number_of_sides = extracted_sides;
			
			if ( s3 == end ) // at end; no (optional) modifier supplied
				result.m_modifier = 0;
			else {
				auto const &[s4, extracted_modifier] = read_integer( s3, static_cast<std::size_t>(end-s3) ); // TODO: ensure constraints (>0?)
				if ( s4 != s3+1 ) [[likely]]
					result.m_modifier = extracted_modifier;
				else [[unlikely]]
					throw std::invalid_argument("Invalid format: expected modifier!");
				
				if ( s4 != end ) [[unlikely]]
					throw std::invalid_argument("Invalid format: expected end-of-string!");
			}
			
			return result;
		} // end-of-function rng::dice_distribution::from_dsl
		[[nodiscard]] I number_of_dice()    const noexcept { return m_number_of_dice;  }
		[[nodiscard]] I number_of_sides()   const noexcept { return m_number_of_sides; }
		[[nodiscard]] I modifier()          const noexcept { return m_modifier;        }
		void number_of_dice(  I const new_value ) noexcept { m_number_of_dice  = new_value; }
		void number_of_sides( I const new_value ) noexcept { m_number_of_sides = new_value; }
		void modifier(        I const new_value ) noexcept { m_modifier        = new_value; }
		[[nodiscard]] I min()     const noexcept override { return m_number_of_dice + m_modifier; }
		[[nodiscard]] I max()     const noexcept override { return m_number_of_dice * m_number_of_sides + m_modifier; }
		[[nodiscard]] F average() const noexcept override { return F(.5) * (m_number_of_sides + 1) * m_number_of_dice + m_modifier; }
		[[nodiscard]] /*inline?*/ std::string to_string() const noexcept override {
			std::string result { "Dice Distribution (" };
			if ( m_number_of_sides > 0 )
				result += std::to_string(m_number_of_dice);
			result += 'd' + std::to_string(m_number_of_sides);
			if ( m_modifier != 0 )
				result += (m_modifier > 0 ? "+" : "") + std::to_string(m_modifier) + ')';
			return result;
		}
	private:
		[[nodiscard]] I get( engine &&e ) const noexcept override {
			// TODO: optimize!
			auto result = m_modifier;
			for ( I i=0; i<m_number_of_dice; ++i )
				result += rng::get<I>( I(0), m_number_of_sides, FALK_FWD(e) ) + 1;
			return result;
		}
	};
	
	template <typename T, std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	class uniform_distribution final : public distribution<T,F> {
	// TODO: contemplate the handling of boundaries...
	protected:
		T m_min;
		T m_max;
	public:
		~uniform_distribution() noexcept final = default;
		constexpr uniform_distribution(
			T const min = std::numeric_limits<T>::min(),
			T const max = std::numeric_limits<T>::max()
		) noexcept:
			distribution<T,F> {     },
			m_min             { min },
			m_max             { max }
		{}
		void min( T const new_value )   noexcept { m_min = new_value; }
		void max( T const new_value )   noexcept { m_max = new_value; }
		[[nodiscard]] T min()     const noexcept override { return m_min; }
		[[nodiscard]] T max()     const noexcept override { return m_max; }
		[[nodiscard]] F average() const noexcept override { return F(.5) * m_min + F(.5) * m_max; }
		[[nodiscard]] std::string to_string() const noexcept override {
			std::string result = "Uniform Distribution (";
			result += std::to_string(m_min) + '~' + std::to_string(m_max) + ')';
			return result;
		}   
	private:
		[[nodiscard]] T get( engine &&e ) const noexcept override { return rng::get( m_min, m_max, FALK_FWD(e) ); }
	};
	
	template <typename T, std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	class normal_distribution : public distribution<T,F> {
	protected:
		F                        m_mean;
		F                        m_standard_deviation;
		std::optional<F> mutable m_maybe_cache;
	public:
		virtual ~normal_distribution() noexcept = default;
		constexpr normal_distribution(
			F const mean,
			F const standard_deviation
		) noexcept:
			distribution<T,F>    {                    },
			m_mean               { mean               },
			m_standard_deviation { standard_deviation },
			m_maybe_cache        { std::nullopt       }
		{}
		[[nodiscard]] F mean()                 const noexcept { return m_mean;                    }
		[[nodiscard]] F standard_deviation()   const noexcept { return m_standard_deviation;      }
		void mean(               F const new_value ) noexcept { m_mean               = new_value; }
		void standard_deviation( F const new_value ) noexcept { m_standard_deviation = new_value; }
		[[nodiscard]] inline virtual F maybe_clamp( F const value ) const noexcept = 0; // abstract
		[[nodiscard]] F average()                                   const noexcept override final { return m_mean; }
	private:
		[[nodiscard]] T get( engine &&e ) const noexcept override final {
			F constexpr epsilon = std::numeric_limits<F>::epsilon();
			F constexpr pi_x2   = F(2.) * std::numbers::pi_v<F>;
			
			F result;
			
			if ( m_maybe_cache )
				result = std::exchange( m_maybe_cache, std::nullopt ).value(); // extract & invalidate old value
			else {
				F uniform1, uniform2;
				do {
					uniform1 = rng::random( FALK_FWD(e) );
					uniform2 = rng::random( FALK_FWD(e) );
				}
				while ( epsilon >= uniform1 );
				// use the Box-Muller transform to convert two (.0~1.) uniform values to two normal distribution values
				F const magnitude = std::sqrt( std::log(uniform1) * F(-2.) ) * m_standard_deviation;
				m_maybe_cache     = magnitude * std::cos( pi_x2 * uniform2 ) + m_mean; // cache one of the values
				result            = magnitude * std::sin( pi_x2 * uniform2 ) + m_mean; // make the other our result
			}
			
			if constexpr ( std::floating_point<T> )
				return static_cast<T>( maybe_clamp(result) );
			else // round if integral
				return static_cast<T>( std::round( maybe_clamp(result) ) );
		}
	};
	
	template <typename            T = default_precision_floating_point_t,
	          std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	class bounded_normal_distribution final : public normal_distribution<T,F> {
		F m_bound_factor;
		F m_min;
		F m_max;
	public:
		~bounded_normal_distribution() noexcept final = default;
		constexpr bounded_normal_distribution(
			F const mean,
			F const standard_deviation,
			F const bound_factor
		) noexcept:
			normal_distribution<T,F> { mean, standard_deviation }
		{
			this->bound_factor( bound_factor ); // calling mutator
		}
		[[nodiscard]] F bound_factor() const noexcept { return m_bound_factor; }
		void constexpr bound_factor( F const new_value ) noexcept {
			m_bound_factor      = new_value;
			m_min               = this->m_mean - this->m_standard_deviation * new_value;
			m_max               = this->m_mean + this->m_standard_deviation * new_value;
			this->m_maybe_cache = {}; // invalidate cache (if any)
		}
		[[nodiscard]] T min() const noexcept override { return m_min; }
		[[nodiscard]] T max() const noexcept override { return m_max; }
		[[nodiscard]] inline F maybe_clamp( F const value ) const noexcept override { return std::clamp( value, m_min, m_max ); }
		[[nodiscard]] std::string to_string() const noexcept override {
			char buffer[192];
			std::sprintf( buffer, "Bounded Normal Distribution (μ: %g, σ: %g, β: %g)", this->m_mean, this->m_standard_deviation, m_bound_factor );
			return std::string( buffer );
		}
	};
	
	template <typename T, std::floating_point F = default_precision_floating_point_t>
	requires( std::integral<T> or std::floating_point<T> )
	class unbounded_normal_distribution final : public normal_distribution<T,F> {
	public:
		~unbounded_normal_distribution() noexcept final = default;
		constexpr unbounded_normal_distribution(
			F const mean,
			F const standard_deviation
		) noexcept:
			normal_distribution<T,F> { mean, standard_deviation }
		{}
		[[nodiscard]] T constexpr min() const noexcept override { return T(-std::numeric_limits<F>::infinity()); } // ideally static...
		[[nodiscard]] T constexpr max() const noexcept override { return T(+std::numeric_limits<F>::infinity()); } // ideally static...
		[[nodiscard]] F constexpr maybe_clamp( F const value ) const noexcept override { return value; }
		[[nodiscard]] std::string to_string() const noexcept override {
			char buffer[128];
			std::sprintf( buffer, "Unbounded Normal Distribution (μ: %g, σ: %g)", this->m_mean, this->m_standard_deviation );
			return std::string( buffer );
		}
	};
	
	namespace literals {
		# ifndef RNG_DICE_LITERAL
		#    define RNG_DICE_LITERAL d
		# endif
		[[nodiscard]] constexpr auto FALK_CONCATENATE( operator""_ , RNG_DICE_LITERAL ) ( char const * const s, std::size_t const n ) {
			return dice_distribution<default_width_signed_integral_t>::from_dsl( s, n );
		}
	} // end-of-namespace rng::literals
	
	// returns a random value from within some provided distribution (e.g. uniform, normal, dice...)
	template <typename...Args> [[nodiscard]] auto constexpr get( distribution<Args...> const &d, RNG_ENGINE(e) ) noexcept( noexcept( d(e) ) ) {
		return d(e);
	}
	
	// returns a random element in a sized range r
	template <std::ranges::sized_range R>
	[[nodiscard]] auto constexpr element( R &&r, RNG_ENGINE(e) ) -> decltype( *std::ranges::begin(r) ) { // TODO: noexcept
		using namespace std::ranges;
		return *( begin(r) + rng::get<range_size_t<R>>( 0, size(r), e ) );
	}
	
	// === === === === === === === === === [ distribution module end ] === === === === === === === === === //
} // end-of-namespace rng
#undef FALK_FWD
#undef FALK_CONCATENATE
#undef FALK_CONCATENATE_IMPL
