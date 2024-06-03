# TODO: more env vars, make install

DEBUG=1

ALIB=libteensygrad.a
OBJDIR=./obj/
SRCDIR=./src/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-O3
LDFLAGS= 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall

# TODO: compile multiple binaries for multiple examples
EXAMPLE=./examples/tensor_example.c
EXEC=example
EXECDIR=./build

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ=graph.o op.o optimizers.o shape.o tensor.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = Makefile include/teensygrad.h

all: $(EXECDIR) $(OBJDIR) $(ALIB) $(EXEC)

$(EXEC): $(ALIB) $(OBJDIR) $(EXAMPLE)
	$(CC) $(COMMON) $(CFLAGS) $(EXAMPLE) -o $(EXECDIR)/$@ $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(OBJDIR)%.o: $(SRCDIR)%.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	@mkdir -p $@

$(EXECDIR):
	@mkdir -p $@

.PHONY: clean
clean:
	@rm -rf $(ALIB) $(OBJDIR) $(EXECDIR)
