# TODO: more env vars, make install

DEBUG=0

ALIB=libteensygrad.a
OBJDIR=obj/
SRCDIR=src/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-O3
LDFLAGS= 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall

EXAMPLE=examples/tensor_example.c
EXEC=example

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ=graph.o op.o optimizers.o shape.o tensor.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = Makefile include/teensygrad.h

all: $(OBJDIR) $(ALIB) $(EXEC)

$(EXEC): $(ALIB) $(OBJDIR) $(EXAMPLE)
	$(CC) $(COMMON) $(CFLAGS) $(EXAMPLE) -o $@ $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(OBJDIR)%.o: $(SRCDIR)%.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	@mkdir -p $(OBJDIR)

.PHONY: clean
clean:
	@rm -rf $(ALIB) $(OBJDIR) $(EXEC)
