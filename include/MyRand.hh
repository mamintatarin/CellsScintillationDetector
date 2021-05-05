#include <random>
using namespace std;
class MyRandom
{
private:
    // Types
    typedef minstd_rand                             Engine;
    typedef uniform_real_distribution<float>        Distribution;
    typedef variate_generator<Engine, Distribution> Generator;

    // To hold the generator
    Generator* _gen;

public:
    MyRandom( int seed, float minVal, float maxVal )
    {
        _gen = new Generator( Engine( seed ), Distribution( minVal, maxVal ) );
    }

    ~MyRandom()
    {
        delete _gen;
    }

    float next()
    {
        return (*_gen)();
    }
};